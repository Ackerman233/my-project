# 官方 SkillOpt 依赖

本项目使用 [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) 的 ALFWorld 环境与进化基础设施。
官方源码没有复制进本仓库，也没有随此提交修改。

服务器通过绝对路径符号链接引用官方仓库；公开版本用本说明替代该不可移植链接。
首次安装时，在项目根目录将此占位目录改名，再克隆官方仓库：

```bash
mv skillopt skillopt-reference
git clone https://github.com/microsoft/SkillOpt.git skillopt
python -m pip install -e './skillopt[alfworld]'
python -m pip install -e . pytest
```

ALFWorld 数据单独安装，并设置 `ALFWORLD_DATA`。API 凭据使用本机环境变量，不写入仓库。
