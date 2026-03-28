#!/usr/bin/env bash
# 功能：执行任意命令，结束后通过 Pushover 发送紧急提醒
set -euo pipefail

JOB_NAME="${JOB_NAME:-remote_job}"

notify() {
  local code=$1
  local host
  host=$(hostname)

  local title msg
  if [ "$code" -eq 0 ]; then
    title="${JOB_NAME} 完成"
    msg="服务器 ${host} 上的任务已完成"
  else
    title="${JOB_NAME} 失败"
    msg="服务器 ${host} 上的任务失败，退出码=${code}"
  fi

  curl -s \
    --form-string "token=anc6wonaba9q5gsqs8zb6gutq2m3mb" \
    --form-string "user=uiuaop7n519stktwrhdtgjsk2mftgu" \
    --form-string "title=${title}" \
    --form-string "message=${msg}" \
    --form-string "priority=2" \
    --form-string "retry=30" \
    --form-string "expire=3600" \
    --form-string "sound=siren" \
    https://api.pushover.net/1/messages.json >/dev/null || true
}

trap 'notify $?' EXIT

"$@"
