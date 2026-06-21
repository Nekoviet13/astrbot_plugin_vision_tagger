import os
import re
import time
import asyncio
import shutil
import base64
import json
from pathlib import Path
from PIL import Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger


@register(
    "astrbot_plugin_vision_tagger",
    "Nekoviet13",
    "视觉打标插件 - 自动生成训练文件夹并排序",
    "5.2.0"
)
class VisionTagger(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 基础配置
        self.watch_root = self.config.get("watch_root", "D:/Trainning/inbox")
        self.output_base = self.config.get("output_base", "D:/Trainning/training_data")
        self.train_img_base = self.config.get("train_img_base", "D:/Trainning/training_data/img")
        self.train_repeat_count = self.config.get("train_repeat_count", 100)
        self.scan_interval = self.config.get("scan_interval", 10)
        self.notify_after = self.config.get("notify_after", 10)
        self.silent_mode = self.config.get("silent_mode", True)
        self.target_user_id = self.config.get("target_user_id", "1213733068")
        
        # 预处理配置
        self.enable_preprocess = self.config.get("enable_preprocess", True)
        self.target_size = self.config.get("target_size", 512)
        
        self.vision_prompt = self.config.get("vision_prompt", 
            "分析这张图片，用英文输出适合 Stable Diffusion 的提示词。只输出提示词，用逗号分隔。")
        
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # 处理记录
        self.processing_records = {}
        
        # 创建目录
        os.makedirs(self.watch_root, exist_ok=True)
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.train_img_base, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("[视觉打标] 插件已加载")
        logger.info(f"[视觉打标] 监控目录: {self.watch_root}")
        logger.info(f"[视觉打标] 输出目录: {self.output_base}")
        logger.info(f"[视觉打标] 训练图片目录: {self.train_img_base}")
        logger.info(f"[视觉打标] 训练重复次数: {self.train_repeat_count}")
        logger.info("=" * 60)
        
        # 加载已有记录
        self._load_records()
        
        # 启动扫描任务
        self.scan_task = asyncio.create_task(self._scan_loop())
    
    def _load_records(self):
        """加载处理记录"""
        record_file = os.path.join(self.output_base, ".processing_records.json")
        if os.path.exists(record_file):
            try:
                with open(record_file, 'r', encoding='utf-8') as f:
                    self.processing_records = json.load(f)
                logger.info(f"[视觉打标] 已加载记录: {len(self.processing_records)} 个物体")
            except Exception as e:
                logger.error(f"[视觉打标] 加载记录失败: {e}")
    
    def _save_records(self):
        """保存处理记录"""
        record_file = os.path.join(self.output_base, ".processing_records.json")
        try:
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[视觉打标] 保存记录失败: {e}")
    
    async def _send_notification(self, message: str):
        if self.silent_mode:
            logger.info(f"[视觉打标] 通知: {message[:100]}...")
            return
        try:
            session_id = f"aiocqhttp_default:FriendMessage:{self.target_user_id}"
            await self.context.send_message(session_id, message)
        except Exception as e:
            logger.error(f"[视觉打标] 发送失败: {e}")
    
    async def _preprocess_image(self, image_path: Path) -> Path:
        """预处理图片：统一尺寸，保留透明背景"""
        if not self.enable_preprocess:
            return image_path
        
        try:
            img = Image.open(image_path)
            original_mode = img.mode
            
            # 检查是否有透明通道
            has_alpha = img.mode in ('RGBA', 'LA', 'P') or (img.mode == 'P' and 'transparency' in img.info)
            
            if has_alpha:
                # 保留 RGBA 模式，不转换
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # 缩放
                target = (self.target_size, self.target_size)
                img.thumbnail(target, Image.Resampling.LANCZOS)
                
                # 如果需要填充到目标尺寸，用透明色填充
                if img.size != target:
                    new_img = Image.new('RGBA', target, (0, 0, 0, 0))
                    x = (target[0] - img.size[0]) // 2
                    y = (target[1] - img.size[1]) // 2
                    new_img.paste(img, (x, y))
                    img = new_img
            else:
                # 没有透明通道，保持原 RGB 处理
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 缩放
                target = (self.target_size, self.target_size)
                img.thumbnail(target, Image.Resampling.LANCZOS)
                
                # 如果需要填充到目标尺寸，用白色填充
                if img.size != target:
                    new_img = Image.new('RGB', target, (255, 255, 255))
                    x = (target[0] - img.size[0]) // 2
                    y = (target[1] - img.size[1]) // 2
                    new_img.paste(img, (x, y))
                    img = new_img
            
            # 保存为 PNG（保留透明通道）
            new_path = image_path.with_suffix('.png')
            img.save(new_path, 'PNG')
            
            if new_path != image_path:
                os.remove(image_path)
            
            logger.info(f"[视觉打标] 预处理完成: {new_path.name} (模式: {img.mode})")
            return new_path
            
        except Exception as e:
            logger.error(f"[视觉打标] 预处理失败 {image_path.name}: {e}")
            return None
    
    async def _analyze_image(self, image_path: Path, object_name: str) -> str:
        provider = self.context.get_using_provider()
        if not provider:
            return ""
        
        try:
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = image_path.suffix.lower()
            mime = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.webp': 'image/webp'
            }.get(ext, 'image/png')
            image_url = f"data:{mime};base64,{image_base64}"
        except Exception as e:
            logger.error(f"[视觉打标] 读取图片失败: {e}")
            return ""
        
        user_instruction = f"这是一个{object_name}，请识别它的视觉特征。"
        prompt = f"{user_instruction}\n\n{self.vision_prompt}"
        
        try:
            response = await provider.text_chat(
                prompt=prompt,
                system_prompt="",
                contexts=[],
                images=[image_url]
            )
            if response and response.completion_text:
                return response.completion_text.strip()
        except Exception as e:
            logger.error(f"[视觉打标] API 失败: {e}")
        
        return ""
    
    def _get_next_sequence(self, train_folder: Path) -> int:
        max_seq = 0
        for ext in self.supported_extensions:
            for f in train_folder.glob(f"*{ext}"):
                match = re.match(r'^(\d{3})\.', f.name)
                if match:
                    max_seq = max(max_seq, int(match.group(1)))
        return max_seq + 1
    
    async def _process_and_save(self, image_path: Path, object_name: str) -> bool:
        logger.info(f"[视觉打标] 处理: {object_name}/{image_path.name}")
        
        # 预处理
        processed_path = await self._preprocess_image(image_path)
        if not processed_path:
            return False
        
        # 分析图片
        prompt = await self._analyze_image(processed_path, object_name)
        if not prompt:
            if processed_path != image_path:
                os.remove(processed_path)
            return False
        
        # 确定训练文件夹（使用可配置的 train_img_base）
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', object_name).lower()
        train_folder_name = f"{self.train_repeat_count}_{safe_name}"
        train_folder = Path(self.train_img_base) / train_folder_name
        train_folder.mkdir(parents=True, exist_ok=True)
        
        # 获取序号
        next_seq = self._get_next_sequence(train_folder)
        base_name = f"{next_seq:03d}"
        
        # 保存提示词
        txt_path = train_folder / f"{base_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # 保存图片
        img_dest = train_folder / f"{base_name}.png"
        shutil.copy2(processed_path, img_dest)
        
        # 删除临时文件
        if processed_path != image_path:
            os.remove(processed_path)
        
        # 更新记录
        if object_name not in self.processing_records:
            self.processing_records[object_name] = {"count": 0, "last_notify": 0}
        
        self.processing_records[object_name]["count"] += 1
        self._save_records()
        
        # 通知
        current_count = self.processing_records[object_name]["count"]
        last_notify = self.processing_records[object_name].get("last_notify", 0)
        
        if current_count - last_notify >= self.notify_after:
            logger.info(f"[视觉打标] {object_name} 已处理 {current_count} 张")
            await self._send_notification(f"📚 已学习 {object_name}\n累计处理: {current_count} 张图片")
            self.processing_records[object_name]["last_notify"] = current_count
            self._save_records()
        
        logger.info(f"[视觉打标] 已保存: {train_folder}/{base_name}.png")
        return True
    
    async def _scan_loop(self):
        logger.info("[视觉打标] 扫描循环已启动")
        
        while True:
            await asyncio.sleep(self.scan_interval)
            
            try:
                watch_path = Path(self.watch_root)
                if not watch_path.exists():
                    continue
                
                for subdir in watch_path.iterdir():
                    if not subdir.is_dir():
                        continue
                    
                    object_name = subdir.name
                    
                    # 收集图片
                    images = []
                    for ext in self.supported_extensions:
                        images.extend(subdir.glob(f"*{ext}"))
                        images.extend(subdir.glob(f"*{ext.upper()}"))
                    
                    if not images:
                        continue
                    
                    logger.info(f"[视觉打标] 发现物体: {object_name}, {len(images)} 张图片")
                    
                    images.sort(key=lambda x: os.path.getctime(x))
                    
                    for img_path in images:
                        success = await self._process_and_save(img_path, object_name)
                        if success:
                            os.remove(img_path)
                    
                    # 保留文件夹，不删除（用户可以继续放入新图片）
                    # 原删除空文件夹的代码已移除
                    
            except Exception as e:
                logger.error(f"[视觉打标] 扫描出错: {e}")
    
    @filter.command("学习统计")
    async def learning_stats(self, event: AstrMessageEvent):
        if not self.processing_records:
            yield event.plain_result("暂无学习记录。把图片放到 inbox 文件夹就开始学习了。")
            return
        
        result = "📊 学习统计：\n"
        for obj, data in self.processing_records.items():
            result += f"  • {obj}: {data['count']} 张\n"
        yield event.plain_result(result)
    
    @filter.command("学习详情")
    async def learning_detail(self, event: AstrMessageEvent, object_name: str = ""):
        if not object_name:
            yield event.plain_result("请指定物体名，例如：/学习详情 AK74M")
            return
        
        data = self.processing_records.get(object_name)
        if not data:
            yield event.plain_result(f"还没有学习过 {object_name}")
            return
        
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', object_name).lower()
        train_folder = Path(self.train_img_base) / f"{self.train_repeat_count}_{safe_name}"
        
        result = f"📚 {object_name} 学习详情：\n"
        result += f"  已处理图片：{data['count']} 张\n"
        result += f"  训练数据位置：{train_folder}"
        
        yield event.plain_result(result)
    
    async def terminate(self):
        if self.scan_task:
            self.scan_task.cancel()
        self._save_records()
        logger.info("[视觉打标] 插件已卸载")
