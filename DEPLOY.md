1. 确认 python 版本，建议使用 conda 虚拟环境安装 **python3.11**  
（官方文档说是 3.10–3.12，我没试过应该也可以），conda 教程这里就不讲了（并非偷懒，bushi

2. 安装 poetry，不同系统方法不同这里就不细说了。然后在项目根目录`poetry install`

3. 部署 one-api  
👉 [one-api 项目地址](https://github.com/songquanpeng/one-api)

4. 修改入口文件 `main.py (graphrag\graphrag\cli)`，19 行有注释，把你想用的模型写上去就好了

5. 创建仓库
```bash
mkdir -p ./ragtest/input
```

6.初始化仓库
```bash
graphrag init --root ./ragtest
```

7.修改配置 
初始化后仓库里会出现 settings.yaml 文件，把里面的参数改为：
```yaml
type: oneapi-chat
type: oneapi-embedding
api_base: http://你的-one-api-地址  //不需要加v1，如果是本地部署就是http://localhost:端口
# model模型自定义（必须是在第4步打过补丁的，或者项目本身支持的）
```

8.在仓库的 input/ 中放入需要构建索引的文件
（官方文档说支持 txt、csv、json），可以用官方示例

9.构建索引
```bash
graphrag index --root ./ragtest
```

10.查询
```bash
graphrag query \
  --root ./ragtest \
  --method global \
  --query "你的问题?"
```
11.如果生成社区报告有问题，看看日志，可能是格式有问题。
可以在 `prompts/community_report_graph.txt` 和 `prompts/community_report_text.txt` 末尾加上：
```yaml
Return output as a JSON object only.
Do not include code fences, explanations, or extra text.
Output must be valid JSON.
```
以及如果想让图谱语言为中文，也是更改prompt，比如把graph相关prompt里面的English改为Chinese
