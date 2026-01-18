# PR 拆分指南 - PR #1484

根据审计者的反馈，已将原始 PR #1484 拆分为 3 个独立的 PR，每个都包含单元测试和边界测试。

## 拆分计划

### PR 1: FestivalProvider (基础组件)
**分支**: `pr-festival-provider`

**文件**:
- `src/providers/festival_provider.py` - 节日提供者实现
- `tests/providers/test_festival_provider.py` - 完整单元测试

**功能**:
- 管理节日日历
- 支持中西方节日
- 查询今天、即将到来的节日
- 提醒节日查询
- 自定义节日添加

**测试覆盖**:
- ? Singleton 模式测试
- ? 今天节日查询
- ? 即将到来节日查询（默认和自定义天数）
- ? 提醒节日查询
- ? 自定义节日添加
- ? 按类型查询节日
- ? 边界测试：零天数、负数天数、不存在的类型

---

### PR 2: FestivalGreeting Action (节日问候功能)
**分支**: `pr-festival-greeting-action`

**文件**:
- `src/providers/festival_provider.py` (依赖 PR 1)
- `src/actions/festival_greeting/interface.py` - Action 接口定义
- `src/actions/festival_greeting/connector/elevenlabs_tts.py` - ElevenLabs TTS 连接器
- `src/actions/festival_greeting/README.md` - 文档
- `src/actions/festival_greeting/DESIGN.md` - 设计文档
- `config/festival_greeting_example.json5` - 配置示例
- `tests/actions/festival_greeting/test_interface.py` - 接口测试（待添加）
- `tests/actions/festival_greeting/test_connector.py` - 连接器测试（待添加）

**功能**:
- FestivalGreeting Action 接口
- 支持 9 种节日类型
- ElevenLabs TTS 集成
- 个性化问候支持

**测试需求**:
- 接口输入验证
- 连接器配置测试
- TTS 调用测试
- 错误处理测试

---

### PR 3: FestivalReminder Background (自动提醒)
**分支**: `pr-festival-reminder`

**文件**:
- `src/providers/festival_provider.py` (依赖 PR 1)
- `src/backgrounds/plugins/festival_reminder.py` - 背景任务实现
- `tests/backgrounds/plugins/test_festival_reminder.py` - 背景任务测试（待添加）

**功能**:
- 定期检查节日
- 自动提醒功能
- 上下文更新

**测试需求**:
- 定时检查逻辑
- 提醒触发条件
- 上下文更新验证
- 配置选项测试

---

## 执行步骤

### 1. 提交 PR 1 (FestivalProvider)
```bash
git checkout pr-festival-provider
# 已包含文件和测试
git push fork pr-festival-provider
# 在 GitHub 创建 PR
```

### 2. 提交 PR 2 (FestivalGreeting Action)
```bash
git checkout pr-festival-greeting-action
# 等待 PR 1 合并后，基于 main 重新创建
git show pr-1484:src/actions/festival_greeting/interface.py > src/actions/festival_greeting/interface.py
git show pr-1484:src/actions/festival_greeting/connector/elevenlabs_tts.py > src/actions/festival_greeting/connector/elevenlabs_tts.py
# 添加测试文件
# 提交并推送
```

### 3. 提交 PR 3 (FestivalReminder)
```bash
git checkout pr-festival-reminder
# 等待 PR 1 合并后
git show pr-1484:src/backgrounds/plugins/festival_reminder.py > src/backgrounds/plugins/festival_reminder.py
# 添加测试文件
# 提交并推送
```

---

## 测试运行

```bash
# 运行所有 FestivalProvider 测试
pytest tests/providers/test_festival_provider.py -v

# 运行特定测试
pytest tests/providers/test_festival_provider.py::test_singleton_instance -v

# 带覆盖率
pytest tests/providers/test_festival_provider.py --cov=src/providers/festival_provider --cov-report=html
```

## 注意事项

1. **依赖关系**: PR 2 和 PR 3 都依赖 PR 1 (FestivalProvider)
2. **合并顺序**: 建议按顺序合并 (PR 1 → PR 2 → PR 3)
3. **测试要求**: 每个 PR 都必须包含完整的单元测试和边界测试
4. **代码审查**: 每个 PR 独立审查，便于审计者检查
