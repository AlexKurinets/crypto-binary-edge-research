-- market: what the recorder captured. research: what each walk-forward run concluded.
-- paper: the forward log, written at decision time and settled later.

create schema if not exists market;
create schema if not exists research;
create schema if not exists paper;

create table if not exists market.meta (
  key   text primary key,
  value jsonb not null
);

create table if not exists market.bars (
  asset text not null,
  ts    timestamptz not null,               -- bar close time; the bar covers the minute before it
  open  double precision not null,
  high  double precision not null,
  low   double precision not null,
  close double precision not null,
  primary key (asset, ts)
);

create table if not exists market.releases (
  ts   timestamptz primary key,
  name text not null
);

create table if not exists market.contracts (
  ticker     text primary key,
  asset      text not null,
  open_ts    timestamptz not null,
  close_ts   timestamptz not null,
  strike     double precision not null,
  settle_avg double precision not null,     -- 60-second average of the index before close_ts
  result     boolean not null               -- settle_avg >= strike
);
create index if not exists contracts_close_ts on market.contracts (close_ts);

create table if not exists market.quotes (
  ticker       text not null references market.contracts,
  ts           timestamptz not null,        -- decision time
  yes_bid      smallint not null check (yes_bid between 1 and 99),
  yes_ask      smallint not null check (yes_ask between 1 and 99 and yes_ask > yes_bid),
  bid_size     integer not null,
  ask_size     integer not null,
  fill_ts      timestamptz not null,        -- when an order sent at ts reaches the book
  fill_yes_bid smallint not null,
  fill_yes_ask smallint not null,
  primary key (ticker, ts)
);
create index if not exists quotes_ts on market.quotes (ts);

create table if not exists research.registrations (
  hypotheses_hash text primary key,
  first_run_at    timestamptz not null default now(),
  hypothesis_ids  text[] not null
);

create table if not exists research.runs (
  run_id          serial primary key,
  created_at      timestamptz not null default now(),
  research_start  timestamptz not null,
  cutoff          timestamptz not null,
  config          jsonb not null,
  hypotheses_hash text not null references research.registrations,
  summary         jsonb not null default '{}'
);

create table if not exists research.folds (
  run_id      int not null references research.runs on delete cascade,
  fold        int not null,
  train_start timestamptz not null,
  train_end   timestamptz not null,
  test_start  timestamptz not null,
  test_end    timestamptz not null,
  model       jsonb not null,
  primary key (run_id, fold)
);

create table if not exists research.predictions (
  run_id  int not null references research.runs on delete cascade,
  fold    int not null,
  ticker  text not null,
  ts      timestamptz not null,
  asset   text not null,
  tau_min int not null,
  p_raw   double precision not null,
  p_model double precision not null,
  mid     double precision not null,
  outcome boolean not null,
  primary key (run_id, ticker, ts)
);

create table if not exists research.hypotheses (
  run_id        int not null references research.runs on delete cascade,
  hypothesis_id text not null,
  name          text not null,
  family        text not null,
  description   text not null,
  signals       int not null,
  excluded      int not null,                -- signals blocked by the hypothesis's own no-trade filters
  n_trades      int not null,
  n_clusters    int not null,
  contracts     int not null,
  fill_rate     double precision,
  hit_rate      double precision,
  gross         double precision,
  spread        double precision,
  latency       double precision,
  fee           double precision,
  net           double precision,
  net_ci_lo     double precision,
  net_ci_hi     double precision,
  gross_ci_lo   double precision,
  gross_ci_hi   double precision,
  p_value       double precision,
  q_value       double precision,
  verdict       text not null,
  pnl_usd       double precision not null,
  primary key (run_id, hypothesis_id)
);

create table if not exists research.trades (
  run_id         int not null references research.runs on delete cascade,
  hypothesis_id  text not null,
  ticker         text not null,
  asset          text not null,
  decided_at     timestamptz not null,
  cluster        text not null,
  side           text not null check (side in ('yes', 'no')),
  decision_price double precision not null,
  fill_price     double precision not null,
  contracts      int not null,
  fee            double precision not null,
  payout         smallint not null,
  gross          double precision not null,
  spread         double precision not null,
  latency        double precision not null,
  fee_per        double precision not null,
  net            double precision not null,
  pnl_usd        double precision not null,
  excluded       boolean not null,           -- blocked by a registered no-trade filter; settled but not counted
  primary key (run_id, hypothesis_id, ticker, decided_at)
);

create table if not exists research.conditions (
  run_id        int not null references research.runs on delete cascade,
  hypothesis_id text not null,
  dimension     text not null,
  bucket        text not null,
  sort_order    int not null,
  n_trades      int not null,
  n_clusters    int not null,
  net           double precision,
  ci_lo         double precision,
  ci_hi         double precision,
  verdict       text not null check (verdict in ('keep', 'avoid', 'within noise', 'too few')),
  primary key (run_id, hypothesis_id, dimension, bucket)
);

create table if not exists research.calibration (
  run_id int not null references research.runs on delete cascade,
  period text not null check (period in ('research', 'paper')),
  source text not null check (source in ('model', 'market')),
  bin    int not null,
  p_mean double precision not null,
  freq   double precision not null,
  n      int not null,
  primary key (run_id, period, source, bin)
);

create table if not exists research.frozen_rules (
  run_id        int primary key references research.runs on delete cascade,
  hypothesis_id text,                       -- null when nothing survived
  params        jsonb,
  avoid         jsonb not null default '[]',
  model         jsonb not null,
  frozen_at     timestamptz not null,
  rule_hash     text not null
);

create table if not exists paper.decisions (
  decision_id bigserial primary key,
  run_id      int not null references research.runs on delete cascade,
  decided_at  timestamptz not null,
  logged_at   timestamptz not null,
  asset       text not null,
  ticker      text not null,
  strike      double precision not null,
  tau_min     int not null,
  index_price double precision not null,
  p_model     double precision not null,
  yes_bid     smallint not null,
  yes_ask     smallint not null,
  side        text not null check (side in ('yes', 'no')),
  price       double precision not null,
  edge        double precision not null,
  regime      text not null,
  action      text not null check (action in ('filled', 'missed', 'skipped')),
  skip_reason text,
  fill_price  double precision,
  contracts   int not null default 0,
  fee         double precision not null default 0,
  settled_at  timestamptz,
  outcome     boolean,
  payout      smallint,
  net         double precision,
  pnl_usd     double precision,
  check (logged_at >= decided_at),
  check (settled_at is null or settled_at > logged_at)
);
create index if not exists decisions_run on paper.decisions (run_id, decided_at);

create table if not exists paper.predictions (
  run_id     int not null references research.runs on delete cascade,
  ticker     text not null,
  decided_at timestamptz not null,
  asset      text not null,
  tau_min    int not null,
  p_model    double precision not null,
  mid        double precision not null,
  outcome    boolean,
  primary key (run_id, ticker, decided_at)
);

-- A logged prediction is evidence only if it cannot be edited once the outcome is known.
-- Settlement may fill in the outcome columns; everything decided at logged_at is frozen.
create or replace function paper.freeze_decision() returns trigger language plpgsql as $$
begin
  if (new.decided_at, new.logged_at, new.ticker, new.p_model, new.yes_bid, new.yes_ask, new.side,
      new.price, new.edge, new.action, new.skip_reason, new.fill_price, new.contracts, new.fee)
     is distinct from
     (old.decided_at, old.logged_at, old.ticker, old.p_model, old.yes_bid, old.yes_ask, old.side,
      old.price, old.edge, old.action, old.skip_reason, old.fill_price, old.contracts, old.fee) then
    raise exception 'paper decision % was logged at % and cannot be changed', old.decision_id, old.logged_at;
  end if;
  if old.settled_at is not null then
    raise exception 'paper decision % is already settled', old.decision_id;
  end if;
  return new;
end $$;

drop trigger if exists freeze_decision on paper.decisions;
create trigger freeze_decision before update on paper.decisions
  for each row execute function paper.freeze_decision();
