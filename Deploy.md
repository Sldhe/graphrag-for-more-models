1.确认python版本，建议使用conda虚拟环境安装python3.11（官方文档说是3.10-3.12，我没试过应该也可以），conda教程这里就不讲了（并非偷懒，bushi

2.安装poetry，不同系统方法不同这里就不细说了

3.部署one-api（https://github.com/songquanpeng/one-api）

4. 修改入口文件main.py(graphrag\graphrag\cli)，19行有注释，把你想用的模型写上去就好了

5. 接下来就是创建仓库（mkdir -p ./ragtest/input）

6. 初始化仓库（graphrag init --root ./ragtest）

7. 修改配置文件。初始化后仓库里会出现settings.yaml文件，把default_chat_model和default_embedding_model的type改为oneapi-chat和oneapi-embedding，
然后api_base改成one-api的地址，模型自定义（必须是在第4步打过补丁的或者项目本身支持的）

8. 在仓库的input中放入需要构建索引的文件（官方文档说支持txt，csv，json），可以用官方文档示例

9. 构建索引index（graphrag index --root ./ragtest）

10. 搜索，使用

graphrag query \
--root ./ragtest \
--method global \
--query "你的问题?"

（Windows Terminal不支持\，可以用`或写成一行）

11. 如果生成社区报告有问题的话看看日志，可能是格式有问题，可以在仓库/prompts里的community_report_graph.txt和community_report_text.txt后面加上
Return output as a JSON object only.
Do not include code fences, explanations, or extra text.
Output must be valid JSON.
