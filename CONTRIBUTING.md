# OpenMind OM1 贡献说明

## 提交内容
新增示例：`examples/weather_agent.py`  
功能：演示如何构建一个查询天气的 Agent  
说明：用户输入城市名后，返回实时天气与气温。

## 改动类型
- [x] 新增示例
- [ ] 修复 bug
- [ ] 文档更新
- [ ] API 改进建议

## 测试结果
已在 Ubuntu 22.04 + Python 3.10 环境下通过测试。  
执行命令：
```bash
uv run examples/weather_agent.py
```
输入“上海”，输出预期天气信息。

## 额外说明
该示例对新开发者非常友好，可作为入门模板。
