"""
别样觉醒 — 数字人形象生成器
对接 LiveTalking 的 Avatar 生成 API，从视频/照片创建数字人形象。

融合自 lipku/LiveTalking avatars/ 模块的生成流程：
  上传视频/照片 → 人脸检测 → 特征提取 → 生成数字人形象 → 注册到 LiveTalking

支持：
  - 从视频生成（推荐：一段1-5分钟说话视频）
  - 从照片生成（单张正面照，效果略差）
  - 形象管理（列表、删除）

依赖：
  - LiveTalking 服务运行中
  - requests 库

作者：AtomCollide-智械工坊
融合来源：lipku/LiveTalking avatars/genavatar.py
"""

import os
import time
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, List


# ── 配置 ──────────────────────────────────────────────────────────────────────

LIVETALKING_DEFAULT_HOST = "http://localhost:8010"


@dataclass
class AvatarBuildResult:
    """形象构建结果"""
    task_id: str = ""
    avatar_id: str = ""
    status: str = ""           # pending / processing / done / failed
    progress: float = 0.0      # 0-100
    message: str = ""
    output_path: str = ""


# ── Avatar 生成客户端 ─────────────────────────────────────────────────────────

class AvatarBuilder:
    """
    数字人形象生成器。

    对接 LiveTalking 的 /avatar/* API，从视频或照片生成数字人形象。

    用法：
        builder = AvatarBuilder("http://localhost:8010")

        # 从视频生成（推荐）
        result = builder.build_from_video("/path/to/talking_video.mp4", "my_avatar")

        # 从照片生成
        result = builder.build_from_photo("/path/to/photo.jpg", "my_avatar")

        # 等待完成
        builder.wait_for_completion(result.task_id)
    """

    def __init__(self, host: str = LIVETALKING_DEFAULT_HOST, timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """检查 LiveTalking 服务是否可用"""
        try:
            resp = requests.get(f"{self.host}/", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def build_from_video(
        self,
        video_path: str,
        avatar_id: str,
        model: str = "wav2lip",
    ) -> AvatarBuildResult:
        """
        从视频生成数字人形象。

        推荐：1-5分钟正面说话视频，光线均匀，背景简洁。

        Args:
            video_path: 视频文件路径
            avatar_id: 形象ID（英文，用于后续引用）
            model: 数字人模型 (wav2lip/musetalk/ultralight)

        Returns:
            AvatarBuildResult 包含 task_id 和状态
        """
        if not os.path.exists(video_path):
            return AvatarBuildResult(status="failed", message=f"视频不存在: {video_path}")

        if not self.health_check():
            return AvatarBuildResult(
                status="failed",
                message=f"LiveTalking 服务未运行: {self.host}",
            )

        print(f"🎬 正在从视频生成数字人形象...")
        print(f"   视频: {video_path}")
        print(f"   形象ID: {avatar_id}")
        print(f"   模型: {model}")

        try:
            with open(video_path, "rb") as f:
                resp = requests.post(
                    f"{self.host}/avatar/create",
                    files={"video": f},
                    data={
                        "avatar_id": avatar_id,
                        "model": model,
                    },
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            data = resp.json()

            return AvatarBuildResult(
                task_id=data.get("data", {}).get("task_id", ""),
                avatar_id=avatar_id,
                status="processing",
                message="形象生成任务已提交",
            )

        except requests.exceptions.HTTPError as e:
            # 如果API不存在（LiveTalking版本不支持），提供手动方案
            if resp.status_code == 404:
                return self._fallback_manual_guide(video_path, avatar_id, model)
            return AvatarBuildResult(status="failed", message=f"API错误: {e}")
        except Exception as e:
            return AvatarBuildResult(status="failed", message=f"生成失败: {e}")

    def build_from_photo(
        self,
        photo_path: str,
        avatar_id: str,
        model: str = "wav2lip",
    ) -> AvatarBuildResult:
        """
        从照片生成数字人形象。

        效果不如视频，但更简单。适合快速原型。

        Args:
            photo_path: 照片文件路径
            avatar_id: 形象ID
            model: 数字人模型
        """
        if not os.path.exists(photo_path):
            return AvatarBuildResult(status="failed", message=f"照片不存在: {photo_path}")

        print(f"📸 正在从照片生成数字人形象...")
        print(f"   照片: {photo_path}")
        print(f"   形象ID: {avatar_id}")

        try:
            with open(photo_path, "rb") as f:
                resp = requests.post(
                    f"{self.host}/avatar/create",
                    files={"image": f},
                    data={
                        "avatar_id": avatar_id,
                        "model": model,
                    },
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            data = resp.json()

            return AvatarBuildResult(
                task_id=data.get("data", {}).get("task_id", ""),
                avatar_id=avatar_id,
                status="processing",
                message="形象生成任务已提交",
            )
        except Exception as e:
            return self._fallback_manual_guide(photo_path, avatar_id, model)

    def _fallback_manual_guide(
        self, media_path: str, avatar_id: str, model: str
    ) -> AvatarBuildResult:
        """
        当 API 不可用时，提供手动操作指南。

        LiveTalking 的 avatar 生成也可以通过 Web 页面操作：
        访问 http://host:8010/avatar.html
        """
        guide = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 手动生成数字人形象指南
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 打开 LiveTalking Avatar 生成页面:
   {self.host}/avatar.html

2. 上传素材: {media_path}

3. 设置形象ID: {avatar_id}

4. 选择模型: {model}

5. 点击"生成"，等待完成

或使用命令行:
   cd /path/to/LiveTalking
   python app.py --model {model} --avatar_id {avatar_id}

生成后的形象文件位于:
   data/avatars/{avatar_id}/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(guide)
        return AvatarBuildResult(
            avatar_id=avatar_id,
            status="manual",
            message=guide.strip(),
        )

    def wait_for_completion(
        self,
        task_id: str,
        max_wait: int = 600,
        poll_interval: int = 5,
    ) -> AvatarBuildResult:
        """等待形象生成完成"""
        if not task_id:
            return AvatarBuildResult(status="unknown", message="无task_id")

        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = requests.get(
                    f"{self.host}/avatar/status/{task_id}",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)

                print(f"  ⏳ {status} ({progress:.0f}%)")

                if status == "done":
                    return AvatarBuildResult(
                        task_id=task_id,
                        avatar_id=data.get("avatar_id", ""),
                        status="done",
                        progress=100,
                        message="形象生成完成",
                    )
                elif status == "failed":
                    return AvatarBuildResult(
                        task_id=task_id,
                        status="failed",
                        message=data.get("error", "生成失败"),
                    )

            except Exception:
                pass

            time.sleep(poll_interval)

        return AvatarBuildResult(
            task_id=task_id,
            status="timeout",
            message=f"等待超时({max_wait}秒)",
        )

    def list_avatars(self) -> List[Dict]:
        """列出已有的数字人形象"""
        try:
            resp = requests.get(f"{self.host}/avatar/list", timeout=10)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception:
            return []

    def delete_avatar(self, avatar_id: str) -> bool:
        """删除数字人形象"""
        try:
            resp = requests.post(
                f"{self.host}/avatar/delete",
                json={"avatar_id": avatar_id},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False


# ── 形象素材建议 ──────────────────────────────────────────────────────────────

MATERIAL_GUIDE = """
📹 数字人形象素材建议

【视频素材（推荐）】
  - 时长：1-5分钟
  - 内容：正面说话，自然表情
  - 光线：均匀柔和，避免强烈阴影
  - 背景：简洁纯色，避免复杂场景
  - 分辨率：720p以上
  - 帧率：25fps以上

【照片素材（快速原型）】
  - 正面照，自然表情
  - 光线均匀
  - 分辨率 512x512 以上
  - 避免墨镜、口罩等遮挡

【模型选择建议】
  - wav2lip：轻量快速，RTX 3060即可，适合大多数场景
  - musetalk：更自然，需RTX 3080Ti+，适合高质量需求
  - ultralight：极轻量，适合移动端/低配GPU
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="别样觉醒 — 数字人形象生成器",
        epilog="""
示例:
  # 从视频生成数字人形象
  python3 avatar_builder.py --video talking.mp4 --avatar-id chenlong

  # 从照片生成
  python3 avatar_builder.py --photo photo.jpg --avatar-id chenlong

  # 查看素材建议
  python3 avatar_builder.py --guide

  # 列出已有形象
  python3 avatar_builder.py --list
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=LIVETALKING_DEFAULT_HOST, help="LiveTalking 地址")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--photo", help="照片文件路径")
    parser.add_argument("--avatar-id", default="default", help="形象ID")
    parser.add_argument("--model", default="wav2lip", choices=["wav2lip", "musetalk", "ultralight"], help="模型")
    parser.add_argument("--guide", action="store_true", help="查看素材建议")
    parser.add_argument("--list", action="store_true", help="列出已有形象")

    args = parser.parse_args()

    if args.guide:
        print(MATERIAL_GUIDE)
        return

    builder = AvatarBuilder(host=args.host)

    if args.list:
        avatars = builder.list_avatars()
        if avatars:
            print(f"\n📋 已有数字人形象 ({len(avatars)} 个):")
            for a in avatars:
                print(f"   - {a.get('avatar_id', '?')}: {a.get('status', '?')}")
        else:
            print("📋 暂无已生成的形象")
        return

    if args.video:
        result = builder.build_from_video(args.video, args.avatar_id, args.model)
    elif args.photo:
        result = builder.build_from_photo(args.photo, args.avatar_id, args.model)
    else:
        parser.print_help()
        return

    print(f"\n{'✅' if result.status == 'done' else '⏳'} 状态: {result.status}")
    if result.message:
        print(f"   {result.message}")


if __name__ == "__main__":
    main()
