#!/bin/bash

# CARLA自动重启脚本 (带Slack通知功能)
# 使用方法: ./carla_auto_restart.sh

# CARLA启动参数
CARLA_CMD="${CARLA_ROOT}/CarlaUE4.sh -quality-level=Epic -world-port=6000 -carla-rpc-port=3000 -RenderOffScreen"

# 检查间隔(秒)
CHECK_INTERVAL=10

# 日志文件
LOG_FILE="carla_restart.log"

# Slack Webhook URL
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"

# 服务器信息
SERVER_NAME=$(hostname)
SERVER_IP=$(hostname -I | awk '{print $1}')
SCRIPT_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# 日志函数
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Slack通知函数
send_slack_notification() {
    local message="$1"
    local emoji="$2"
    local color="$3"
    
    # 构建JSON消息
    local json_message=$(cat <<EOF
{
    "text": "${emoji} CARLA 服务器状态通知",
    "attachments": [
        {
            "color": "${color}",
            "fields": [
                {
                    "title": "服务器信息",
                    "value": "主机: ${SERVER_NAME}\nIP: ${SERVER_IP}\n时间: $(date '+%Y-%m-%d %H:%M:%S')",
                    "short": true
                },
                {
                    "title": "状态详情", 
                    "value": "${message}",
                    "short": true
                }
            ]
        }
    ]
}
EOF
    )
    
    # 发送到Slack
    curl -X POST \
         -H 'Content-type: application/json' \
         --data "$json_message" \
         "$SLACK_WEBHOOK_URL" \
         --silent --show-error >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✅ Slack通知发送成功"
    else
        log "❌ Slack通知发送失败"
    fi
}

# 检查CARLA_ROOT是否设置
if [ ! -d "$CARLA_ROOT" ]; then
    echo -e "${RED}错误: CARLA_ROOT路径不存在: $CARLA_ROOT${NC}"
    echo "请设置正确的CARLA路径:"
    echo "  export CARLA_ROOT=/path/to/your/carla"
    echo "  然后运行: ./carla_auto_restart.sh"
    echo ""
    echo "或者直接修改脚本中的CARLA_ROOT变量"
    exit 1
fi

if [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo -e "${RED}错误: 找不到CarlaUE4.sh文件: ${CARLA_ROOT}/CarlaUE4.sh${NC}"
    exit 1
fi

# 清理函数
cleanup() {
    log "${YELLOW}收到退出信号，正在停止CARLA...${NC}"
    
    # 发送停止通知到Slack
    local uptime=$(( $(date +%s) - $(date -d "$SCRIPT_START_TIME" +%s) ))
    local uptime_formatted=$(printf "%02d小时%02d分钟%02d秒" $((uptime/3600)) $(((uptime%3600)/60)) $((uptime%60)))
    send_slack_notification "CARLA监控脚本手动停止\n运行时长: ${uptime_formatted}\n重启次数: ${restart_count}次" "⏹️" "warning"
    
    if [ -n "$CARLA_PID" ]; then
        kill -TERM "$CARLA_PID" 2>/dev/null
        wait "$CARLA_PID" 2>/dev/null
        log "${GREEN}CARLA进程已停止${NC}"
    fi
    log "${GREEN}监控已停止${NC}"
    exit 0
}

# 设置信号处理
trap cleanup SIGTERM SIGINT

# 启动CARLA进程
start_carla() {
    local is_restart="$1"
    
    log "${GREEN}启动CARLA进程...${NC}"
    log "命令: $CARLA_CMD"
    log "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log "当前工作目录: $(pwd)"
    log "CARLA_ROOT: $CARLA_ROOT"
    
    # 记录系统资源信息
    log "系统资源信息:"
    log "  内存使用: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
    log "  磁盘使用: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
    log "  CPU负载: $(uptime | awk -F'load average:' '{print $2}')"
    
    # 启动CARLA并获取PID
    $CARLA_CMD >> "$LOG_FILE" 2>&1 &
    CARLA_PID=$!
    
    log "进程PID: $CARLA_PID"
    
    # 等待一下确保进程启动
    sleep 5
    
    # 检查进程是否还在运行
    if kill -0 "$CARLA_PID" 2>/dev/null; then
        log "${GREEN}CARLA启动成功 (PID: $CARLA_PID)${NC}"
        
        # 发送启动成功通知到Slack
        if [ "$is_restart" = "restart" ]; then
            send_slack_notification "CARLA服务器重启成功\nPID: ${CARLA_PID}\n重启次数: ${restart_count}次" "🔄" "good"
        else
            send_slack_notification "CARLA服务器启动成功\nPID: ${CARLA_PID}\n启动时间: $(date '+%Y-%m-%d %H:%M:%S')" "🚀" "good"
        fi
        
        return 0
    else
        log "${RED}CARLA启动失败${NC}"
        
        # 发送启动失败通知到Slack
        if [ "$is_restart" = "restart" ]; then
            send_slack_notification "CARLA服务器重启失败\n重启次数: ${restart_count}次\n将继续尝试重启" "❌" "danger"
        else
            send_slack_notification "CARLA服务器初始启动失败\n脚本将退出" "💥" "danger"
        fi
        
        return 1
    fi
}

# 检查CARLA是否运行
is_carla_running() {
    if [ -n "$CARLA_PID" ] && kill -0 "$CARLA_PID" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 主监控循环
log "${GREEN}开始CARLA自动重启监控${NC}"
log "检查间隔: ${CHECK_INTERVAL}秒"
log "脚本启动时间: $SCRIPT_START_TIME"
log "服务器信息: $SERVER_NAME ($SERVER_IP)"

# 初始启动
if ! start_carla "initial"; then
    log "${RED}初始启动失败，退出${NC}"
    exit 1
fi

# 监控循环
restart_count=0
last_health_check=$(date +%s)
health_check_interval=3600  # 1小时发送一次健康检查

while true; do
    sleep "$CHECK_INTERVAL"
    current_time=$(date +%s)
    
    # 定期发送健康检查通知
    if [ $((current_time - last_health_check)) -ge $health_check_interval ]; then
        uptime=$(( current_time - $(date -d "$SCRIPT_START_TIME" +%s) ))
        uptime_formatted=$(printf "%02d小时%02d分钟" $((uptime/3600)) $(((uptime%3600)/60)))
        memory_usage=$(free | grep '^Mem:' | awk '{printf "%.1f%%", $3/$2 * 100.0}')
        
        log "健康检查: CARLA运行正常 (PID: $CARLA_PID)"
        send_slack_notification "CARLA服务器运行正常\n运行时长: ${uptime_formatted}\n内存使用: ${memory_usage}\n重启次数: ${restart_count}次" "💚" "good"
        
        last_health_check=$current_time
    fi
    
    if ! is_carla_running; then
        crash_time=$(date '+%Y-%m-%d %H:%M:%S')
        restart_count=$((restart_count + 1))
        
        log "${YELLOW}检测到CARLA进程终止，正在重启... (第${restart_count}次)${NC}"
        log "崩溃时间: $crash_time"
        log "进程PID: $CARLA_PID (已终止)"
        
        # 记录崩溃信息
        log "崩溃时系统状态:"
        log "  内存使用: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
        log "  磁盘使用: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
        log "  CPU负载: $(uptime | awk -F'load average:' '{print $2}')"
        
        # 发送崩溃通知到Slack
        send_slack_notification "CARLA服务器崩溃检测\n崩溃时间: ${crash_time}\n重启次数: ${restart_count}次\n正在尝试重启..." "💥" "warning"
        
        # 等待一下再重启
        sleep 3
        
        if start_carla "restart"; then
            log "${GREEN}CARLA重启成功${NC}"
        else
            log "${RED}CARLA重启失败，将在${CHECK_INTERVAL}秒后重试${NC}"
        fi
    fi
done