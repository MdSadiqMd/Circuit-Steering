#!/bin/bash
# Start thermal_guard.sh on the process group of the newest "scripts/puget/run_<name>.sh".
# Usage: scripts/puget/guard.sh <name>    (e.g. a19); log: logs_puget/thermal_<name>.log
cd /dev/shm/cs/circuit-steering || exit 1
pid=$(pgrep -nf "^/bin/bash scripts/puget/run_$1.sh")
[ -n "$pid" ] || {
  echo "no run_$1.sh running"
  exit 1
}
pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
setsid nohup scripts/puget/thermal_guard.sh "$pg" "logs_puget/thermal_$1.log" >/dev/null 2>&1 </dev/null &
echo "guard on pgid $pg"
