# 🌟 Vision Tagger Plugin for AstrBot

**AstrBot 视觉打标插件** —— 自动化 LoRA 数据集生成工具。  
监控指定文件夹，识别图片主体，借助视觉大模型生成高质量提示词，自动构建规范化的训练数据集。

---

## ✨ 主要功能

- 🖼️ **自动监控**：监听 `watch_root` 目录，按子文件夹名称识别物体
- 🤖 **AI 打标**：调用 AstrBot 当前视觉模型，为每张图片生成英文提示词（适用于 Stable Diffusion）
- 🧹 **预处理**：统一缩放图片至指定尺寸，自动处理透明背景（保留 RGBA）
- 📁 **目录归档**：按 `{repeat}_{object_name}` 结构自动创建训练文件夹，并生成带序号的 `.txt` + `.png`
- 📊 **学习统计**：支持查询学习进度和详情
- 🔕 **静默模式**：可关闭主动通知，仅记录日志

---

## 📦 安装

将本项目放置于 AstrBot 插件目录下，或通过 `astrbot` 插件管理命令加载。

```bash
# 假设已进入 AstrBot 插件目录
git clone <your-repo-url> astrbot_plugin_vision_tagger