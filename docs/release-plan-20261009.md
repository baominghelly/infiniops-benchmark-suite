# 容器 Preflight + Periodic Worker 统一发布方案（2026-10-09）

## 0. 一句话
把本地已验证的容器自启动 preflight 发布到 7 平台生产 runner，并补齐 A100 / 寒武纪的 periodic worker 缺口，一次性收拢部署一致性。

## 1. 现状核查（2026-10-08 17:40 只读核实，来源 a100 + cambricon28）

| 项 | 现状 | 证据 |
|---|---|---|
| infinibenchctl 远端启动修复 | ✅ 已部署 | /data-aisoft/baoming/infinibench-ops/bin/infinibenchctl mtime 10-08 08:58，含 .bak-05/06 两份备份 |
| 容器 preflight（新修复） | ❌ 未部署 | ensure_benchmark_container 引用只出现在 shared-script-backups/20260923-* 等备份目录，活动 runner 路径无 |
| A100 periodic 拉起 | ❌ 缺失 | 无本机 worker 目录；Hermes 容器（baoming_hermes_agent / infibench-hermes）在跑但无 cron |
| 寒武纪 periodic worker | ❌ 缺失 | cambricon28 无任何 infinibench/worker 相关 cron |
| 其余 5 平台 periodic | ✅ 有 | a100 crontab 有 ascend_remote / hygon_remote（*/5），共享盘 worker 含 metax/moore/iluvatar/hygon/ascend 目录 |
| daily 全局调度 | ✅ | a100 crontab `0 18 * * * scheduled_daily.sh`（每天 18:00 触发） |
| 本地测试 | ✅ | 今日 141 单测通过（来源：今日工作线程 16:03） |

## 2. 发布内容（从 D:\InfiniBench\infiniops-benchmark-suite）
1. scripts/ensure_benchmark_container.sh（容器 preflight 核心）
2. 各平台 run_*_benchmark.sh（接入 preflight 调用）
3. periodic worker 相关脚本/注册配置（A100 + 寒武纪新增）
4. docs 更新（release_runbook / platform_deployment_map 摘要）

## 3. 时序约束（重要）
- **daily run 每天 18:00 触发，单次约 2~2.5 小时（今日寒武纪 13:55→16:15）**
- 发布窗口二选一：**上午 09:30–17:00**（推荐，失败有整天时间处理）或 **当日 run 结束后 ~21:30**
- ❌ 禁止 17:30–21:00 之间发布（会撞上正在跑的 benchmark）

## 4. 发布顺序（风险从低到高）
1. **共享盘 5 平台**（metax / moore / iluvatar / tianshu / hygon）：改共享路径脚本即生效，先灰度 1 个平台
2. **寒武纪 cambricon28**：新增 periodic worker（从 a100 经 ssh 拉起，复用今日修好的 bash -c + disown 模式）
3. **A100 本机**：进 Hermes 容器改 runner + 加宿主 cron，风险最高放最后

## 5. 每平台标准步骤（模板）
```bash
# 1) 备份（带时间戳，保留 7 天）
cp -a <目标目录> <备份根>/preflight-release-$(date +%Y%m%dT%H%M%SZ)
# 2) 替换文件（rsync --checksum）
# 3) 语法检查
bash -n run_*_benchmark.sh && python3 -m py_compile run_platform_worker.py
# 4) 干跑验证（不启动 benchmark）
grep -n 'ensure_benchmark_container' run_*_benchmark.sh
# 5) 记录 deployment journal：时间 / 主机 / 文件 sha256 / 操作人
```

## 6. 部署后验证（当晚 daily run 即真实回归）
- 次日 09:00 检查：benchmark.done、exit_code、CSV 行数与今日基线对比（寒武纪今日基线：1090 成功 / 128 诊断）
- 任一平台失败率显著升高 → 立即回滚该平台，不影响其他平台

## 7. 回滚
```bash
# 单平台回滚（共享盘）
cp -a <备份目录>/. <目标目录>/
# A100 容器内：恢复备份脚本后 docker restart infibench-hermes
# 验证：sha256sum 与 deployment journal 比对
```

## 8. 需要用户决策的 3 个点
1. A100 periodic 用宿主 cron 还是 systemd timer？（建议 cron，与其余平台一致）
2. 寒武纪 periodic 拉起点放 a100（统一管理）还是 cambricon28 本机？（建议 a100，集中可见）
3. 首发灰度平台选哪个？（建议 metax49：今日刚清理过、行为基线新鲜）

## 9. 执行清单（明天照勾）
- [ ] 09:30 决策上面 3 点
- [ ] 10:00 共享盘灰度平台发布 + 验证
- [ ] 11:00 其余 4 个共享盘平台
- [ ] 14:00 寒武纪 periodic worker
- [ ] 16:00 A100（容器内 + 宿主 cron）
- [ ] 17:00 全平台 preflight 存在性核查
- [ ] 次日 09:00 回归对比
