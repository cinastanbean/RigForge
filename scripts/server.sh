#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_MODULE="rigforge.main:app"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
LOG_FILE="${LOG_FILE:-/tmp/rigforge-uvicorn.log}"
PID_FILE="${PID_FILE:-/tmp/rigforge-uvicorn.pid}"
MODE="${MODE:-bg}" # fg|bg

usage() {
  cat <<EOF
Usage:
  $(basename "$0") start [fg|bg]
  $(basename "$0") restart [fg|bg]
  $(basename "$0") stop
  $(basename "$0") status

Env overrides:
  HOST=127.0.0.1 PORT=8000 PYTHONPATH_VALUE=src LOG_FILE=/tmp/... PID_FILE=/tmp/...
EOF
}

kill_old() {
  local pids
  pids="$(pgrep -f "uvicorn ${APP_MODULE}" || true)"
  if [[ -n "${pids}" ]]; then
    echo "Killing old server process(es): ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} || true
    sleep 1
    pids="$(pgrep -f "uvicorn ${APP_MODULE}" || true)"
    if [[ -n "${pids}" ]]; then
      echo "Force killing remaining process(es): ${pids}"
      # shellcheck disable=SC2086
      kill -9 ${pids} || true
    fi
  fi
  rm -f "${PID_FILE}"
}

start_fg() {
  echo "Starting server in foreground: http://${HOST}:${PORT}"
  cd "${ROOT_DIR}"
  exec env PYTHONPATH="${PYTHONPATH_VALUE}" /opt/anaconda3/bin/uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}"
}

start_bg() {
  echo "Starting server in background: http://${HOST}:${PORT}"
  cd "${ROOT_DIR}"
  nohup env PYTHONPATH="${PYTHONPATH_VALUE}" /opt/anaconda3/bin/uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}" > "${LOG_FILE}" 2>&1 < /dev/null &
  echo $! > "${PID_FILE}"
  local ok=0
  for _ in $(seq 1 15); do
    if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 1
  done

  if [[ "${ok}" == "1" ]]; then
    echo "Server started. PID=$(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
    curl -s -o /dev/null --max-time 2 "http://${HOST}:${PORT}/" || true
  else
    echo "Server failed to start. Check log: ${LOG_FILE}"
    if [[ -f "${PID_FILE}" ]]; then
      local pid
      pid="$(cat "${PID_FILE}")"
      ps -p "${pid}" -o pid=,command= || true
    fi
    tail -n 40 "${LOG_FILE}" || true
    exit 1
  fi
}

status() {
  echo "Port check (${PORT}):"
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || true
  echo
  echo "Process check:"
  pgrep -fal "uvicorn ${APP_MODULE}" || true
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "PID file: ${PID_FILE} -> ${pid} (running)"
    else
      echo "PID file: ${PID_FILE} -> ${pid} (stale)"
    fi
  fi
}

ACTION="${1:-restart}"
ARG_MODE="${2:-${MODE}}"

case "${ACTION}" in
  start)
    kill_old
    if [[ "${ARG_MODE}" == "fg" ]]; then
      start_fg
    else
      start_bg
    fi
    ;;
  restart)
    kill_old
    if [[ "${ARG_MODE}" == "fg" ]]; then
      start_fg
    else
      start_bg
    fi
    ;;
  stop)
    kill_old
    echo "Server stopped."
    ;;
  status)
    status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
