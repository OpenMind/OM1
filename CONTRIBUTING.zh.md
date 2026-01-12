# 贡献 OM1 项目

我们欢迎社区的贡献！OM1 是一个开源项目，我们非常感谢你帮助我们不断改进它。无论你是在修复错误、添加新功能、改进文档，还是提出新的想法，你的贡献都非常重要。

在开始贡献之前，请花一点时间阅读以下指南。这将有助于简化流程，并确保大家在同一标准下协作。

所有 PR 必须清楚说明要解决的问题。没有明确问题描述的更改可能会在未审查的情况下被关闭。

贡献方式

报告 Bug  
如果你发现了 Bug，请在 GitHub 上提交 Issue，并包含以下内容：

- 对 Bug 的清晰、简洁描述  
- 复现步骤  
- 你的操作系统和 Python 版本  
- 相关错误信息或堆栈追踪  
- 截图（如适用）

提出新功能建议  
有新的功能想法或改进建议？请在 GitHub 上创建 Issue，并描述你的想法。说明该功能的动机，以及它如何帮助 OM1 的用户。建议在开始实现之前先进行讨论。

改进文档  
良好的文档至关重要。如果你发现文档中有不清晰、不完整或过时的内容，请提交 PR。包括 README、代码注释和其他文档文件。

修复 Bug  
浏览标记为 bug 或 help wanted 的 Issue。如果你打算解决某个问题，请在 Issue 下留言说明。

实现新功能  
查看标记为 enhancement、bounty 或 help wanted 的 Issue。建议在开始开发前先在 Issue 中讨论你的实现思路。

编写测试  
OM1 追求高测试覆盖率。如果你添加了新代码，请同时添加测试。如果发现测试覆盖不足的地方，补充测试也是很好的贡献。

代码审查  
审查 Pull Request 也是一种非常重要的贡献方式，可以帮助提高代码质量。

不支持的内容（Out of Scope）

- 文档翻译（通常不支持多语言版本，除非维护者特别批准）
- 仅格式、变量命名、风格调整的更改
- 纯视觉或微小无意义的修改
- 仅基于个人偏好的重构

贡献流程（Pull Request）

1. Fork 仓库  
点击 OM1 仓库右上角的 Fork 按钮

2. 克隆你的仓库  

git clone https://github.com/<你的用户名>/OM1.git  
cd OM1  

3. 配置开发环境  
参考文档完成环境配置

4. 创建新分支  

git checkout -b your-branch-name  

5. 提交修改  

git commit -m "docs: add Chinese translation for CONTRIBUTING"  

本地测试

安装 pre-commit  

pre-commit install  
pre-commit run --all-files  

生成文档  

chmod +x scripts/mintlify.sh  
./scripts/mintlify.sh  

运行测试  

uv run pytest --log-cli-level=DEBUG -s  

推送代码  

git push origin your-branch-name  

创建 Pull Request

在 PR 描述中写：

This PR adds a Chinese translation of CONTRIBUTING.md as requested in issue #1326.

编码规范

- 遵守 PEP8  
- 编写清晰的 docstrings  
- 使用 pytest  
- 使用类型提示（PEP484）

行为规范

请保持尊重、包容、合作的态度，遵守 GitHub 社区准则。

维护者政策

- 不符合项目目标的 PR 可能被关闭  
- 被关闭的 PR 可能不会收到详细反馈  

获取帮助

- 提交 Issue  
- 在 PR 或 Issue 评论中提问  
- 加入开发者 Telegram 群  

谢谢你为 OM1 做出贡献！
