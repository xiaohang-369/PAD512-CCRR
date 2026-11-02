# main 快进迁移计划（仅计划，未执行）

本次只创建/推送 v4/stable 和 archive/pre-v4-main-20261102，main不移动。先由用户审阅治理报告、完整文件清单、逐项处置表、回归结果和差异，再明确批准稳定提交的完整SHA。特别需审阅VALIDATION.md中的旧Stage B单项失败与隔离SciPy兼容测试限制；Git可快进不等于全部旧工程测试全绿。

## 批准后的步骤

在干净的独立clone/worktree中普通 `git fetch origin`，不改代理/TLS/服务器环境。把 APPROVED_STABLE_SHA 手工设为已经批准的40位SHA，不自动选择一个后来变化的分支tip。核对：

1. origin/main仍等于5e086984aa26cbd00af24326d2b985afa4b3a282；如有人更新则停止并重新审查。
2. origin/archive/pre-v4-main-20261102仍指向该旧main；origin/v4/stable仍指向批准SHA。
3. `git merge-base --is-ancestor origin/main "$APPROVED_STABLE_SHA"` 返回0；核对批准SHA与治理文件清单/测试一致。
4. `git push --dry-run origin "$APPROVED_STABLE_SHA:refs/heads/main"` 只作预演。
5. 用户批准执行后，普通 `git push origin "$APPROVED_STABLE_SHA:refs/heads/main"`。不使用force/force-with-lease，不rebase，不制造替换历史的根提交。若GitHub保护分支规则拒绝直推，停止并采用符合规则、保持批准SHA历史的审阅流程，不绕过保护。
6. 用 `git ls-remote --heads origin refs/heads/main refs/heads/v4/stable refs/heads/archive/pre-v4-main-20261102` 确认main和stable指向批准SHA，归档仍是旧SHA；再确认实验分支不变。

普通push会在并发更新导致非快进时拒绝更新。保留V4、V2、V3和实验分支；不自动合并新身份排序损失。无需先更新旧工作区本地main，因为它有自己的历史状态。

## 回退原则

归档引用与本地完整bundle用于追溯；若之后需要恢复旧行为，另行准备可审查的修正/恢复提交。不得把本计划解释为允许reset远程main、强推或删除新历史。
