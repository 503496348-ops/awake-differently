"""
Awake Differently — Multi-Format Chat Importer
=================================================
Parsers for real chat export formats from WeChat, Telegram, and Feishu.
Converts platform-specific exports into normalized message format for
the identity distillation pipeline.

Supported formats:
- WeChat: HTML exports (from WeChat desktop export tools like WeChatMsg)
- Telegram: JSON export from Telegram Desktop (Export chat history)
- Feishu/Lark: JSON API response format
- Generic CSV: timestamp,sender,text columns

Author: AtomCollide-智械工坊
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

logger = logging.getLogger("awake_differently.chat_importer")


class ChatPlatform(str, Enum):
    """Supported chat platforms."""
    WECHAT = "wechat"
    TELEGRAM = "telegram"
    FEISHU = "feishu"
    GENERIC_CSV = "csv"
    GENERIC_JSON = "json"


@dataclass
class NormalizedMessage:
    """Platform-agnostic message format for the analysis pipeline."""
    sender: str
    text: str
    timestamp: str  # ISO 8601
    msg_type: str = "text"  # text, image, video, file, sticker, link, post
    platform: str = ""
    reply_to: str = ""  # quoted message text (for reply chain analysis)
    media_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "sender": self.sender,
            "text": self.text,
            "timestamp": self.timestamp,
            "msg_type": self.msg_type,
        }
        if self.platform:
            d["platform"] = self.platform
        if self.reply_to:
            d["reply_to"] = self.reply_to
        return d


@dataclass
class ImportResult:
    """Result of a chat import operation."""
    messages: List[NormalizedMessage]
    platform: ChatPlatform
    person_candidates: List[str]  # most active senders
    time_span_days: int
    total_raw_messages: int
    parse_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.messages) > 0

    def summary(self) -> str:
        return (
            f"平台: {self.platform.value}\n"
            f"解析消息: {len(self.messages)}/{self.total_raw_messages}\n"
            f"时间跨度: {self.time_span_days}天\n"
            f"活跃人物: {', '.join(self.person_candidates[:5])}\n"
            f"解析错误: {len(self.parse_errors)}条"
        )


# ─── WeChat HTML Parser ───────────────────────────────────────────────────────

class WeChatHTMLParser:
    """Parse WeChat HTML exports (from tools like WeChatMsg, WechatExporter).

    WeChat HTML exports typically have this structure:
    <div class="message">
      <div class="nickname">Sender Name</div>
      <div class="content">Message text</div>
      <div class="timestamp">2024-01-15 14:30</div>
    </div>

    Handles variations across different export tools.
    """

    # Common timestamp patterns in WeChat exports
    TIMESTAMP_PATTERNS = [
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
        r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})",
        r"(\d{4}年\d{1,2}月\d{1,2}日\s+\d{2}:\d{2})",
    ]

    def parse(self, file_path: str) -> ImportResult:
        """Parse a WeChat HTML export file."""
        path = Path(file_path)
        if not path.exists():
            return ImportResult(
                messages=[], platform=ChatPlatform.WECHAT,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"文件不存在: {file_path}"],
            )

        html_content = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_html(html_content)

    def parse_html(self, html: str) -> ImportResult:
        """Parse WeChat HTML content."""
        messages = []
        errors = []

        # Strategy 1: Look for structured message divs
        # WeChatMsg tool format
        msg_pattern = re.compile(
            r'<div[^>]*class="[^"]*message[^"]*"[^>]*>(.*?)</div>\s*</div>',
            re.DOTALL | re.IGNORECASE,
        )

        # Try multiple parsing strategies
        messages = self._parse_wechatmsg_format(html, errors)
        if not messages:
            messages = self._parse_generic_html_format(html, errors)
        if not messages:
            messages = self._parse_text_format(html, errors)

        # Deduplicate and sort
        seen = set()
        unique_messages = []
        for msg in sorted(messages, key=lambda m: m.timestamp):
            key = (msg.sender, msg.text, msg.timestamp)
            if key not in seen:
                seen.add(key)
                unique_messages.append(msg)

        # Compute stats
        time_span = self._compute_time_span(unique_messages)
        candidates = self._get_person_candidates(unique_messages)

        return ImportResult(
            messages=unique_messages,
            platform=ChatPlatform.WECHAT,
            person_candidates=candidates,
            time_span_days=time_span,
            total_raw_messages=len(messages),
            parse_errors=errors,
        )

    def _parse_wechatmsg_format(self, html: str, errors: list) -> List[NormalizedMessage]:
        """Parse WeChatMsg desktop export format."""
        messages = []

        # Pattern for WeChatMsg HTML: nickname + content + time
        pattern = re.compile(
            r'<div[^>]*class="[^"]*nickname[^"]*"[^>]*>(.*?)</div>.*?'
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>.*?'
            r'<div[^>]*class="[^"]*time[^"]*"[^>]*>(.*?)</div>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in pattern.finditer(html):
            sender = self._clean_html(match.group(1)).strip()
            text = self._clean_html(match.group(2)).strip()
            time_str = self._clean_html(match.group(3)).strip()

            if not sender or not text:
                continue

            timestamp = self._parse_timestamp(time_str)
            msg_type = self._detect_msg_type(text)

            messages.append(NormalizedMessage(
                sender=sender,
                text=text,
                timestamp=timestamp,
                msg_type=msg_type,
                platform="wechat",
            ))

        return messages

    def _parse_generic_html_format(self, html: str, errors: list) -> List[NormalizedMessage]:
        """Parse generic WeChat HTML export format using flexible patterns."""
        messages = []

        # Look for any divs with sender-like and content-like structures
        # Common: <b>Sender</b>: Message <span>Time</span>
        pattern = re.compile(
            r'<(?:b|strong|span)[^>]*>([^<]{1,50})</(?:b|strong|span)>\s*[:：]\s*'
            r'(.*?)(?:<span[^>]*>[^<]*'
            r'(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})[^<]*</span>)?',
            re.DOTALL,
        )

        for match in pattern.finditer(html):
            sender = self._clean_html(match.group(1)).strip()
            text = self._clean_html(match.group(2)).strip()

            if not sender or not text or len(sender) > 30:
                continue

            # Try to extract timestamp from the text
            timestamp = ""
            for pat in self.TIMESTAMP_PATTERNS:
                ts_match = re.search(pat, text)
                if ts_match:
                    timestamp = self._parse_timestamp(ts_match.group(1))
                    text = text[:ts_match.start()].strip()
                    break

            if text:
                messages.append(NormalizedMessage(
                    sender=sender,
                    text=text,
                    timestamp=timestamp,
                    msg_type=self._detect_msg_type(text),
                    platform="wechat",
                ))

        return messages

    def _parse_text_format(self, text: str, errors: list) -> List[NormalizedMessage]:
        """Parse plain text chat log format (fallback)."""
        messages = []
        # Pattern: "Sender: Message" or "Sender：Message" each on a line
        line_pattern = re.compile(
            r'^([^:：\n]{1,30})[:：]\s*(.+)$',
            re.MULTILINE,
        )

        current_timestamp = ""
        for pat in self.TIMESTAMP_PATTERNS:
            pass  # just checking they exist

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if line is a standalone timestamp
            ts_found = False
            for pat in self.TIMESTAMP_PATTERNS:
                ts_match = re.match(r'^' + pat + r'$', line)
                if ts_match:
                    current_timestamp = self._parse_timestamp(ts_match.group(1))
                    ts_found = True
                    break
            if ts_found:
                continue

            match = line_pattern.match(line)
            if match:
                sender = match.group(1).strip()
                msg_text = match.group(2).strip()
                if msg_text:
                    messages.append(NormalizedMessage(
                        sender=sender,
                        text=msg_text,
                        timestamp=current_timestamp,
                        msg_type=self._detect_msg_type(msg_text),
                        platform="wechat",
                    ))

        return messages

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags and decode entities."""
        text = re.sub(r'<[^>]+>', '', html)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
        return text.strip()

    def _parse_timestamp(self, time_str: str) -> str:
        """Parse various timestamp formats to ISO 8601."""
        time_str = time_str.strip()
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue

        # Chinese format: 2024年1月15日 14:30
        cn_match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})', time_str)
        if cn_match:
            dt = datetime(
                int(cn_match.group(1)), int(cn_match.group(2)), int(cn_match.group(3)),
                int(cn_match.group(4)), int(cn_match.group(5)),
            )
            return dt.isoformat()

        return time_str

    def _detect_msg_type(self, text: str) -> str:
        """Detect message type from content."""
        if text.startswith(('[图片]', '[image]', '[Photo]')):
            return "image"
        if text.startswith(('[视频]', '[video]', '[Video]')):
            return "video"
        if text.startswith(('[文件]', '[file]', '[File]')):
            return "file"
        if text.startswith(('[语音]', '[voice]', '[Voice]')):
            return "audio"
        if text.startswith(('[动画]', '[sticker]', '[Sticker]')):
            return "sticker"
        if text.startswith(('[链接]', '[link]', '[Link]')):
            return "link"
        if re.search(r'https?://\S+', text):
            return "link"
        return "text"

    def _compute_time_span(self, messages: List[NormalizedMessage]) -> int:
        """Compute time span in days."""
        timestamps = []
        for msg in messages:
            if msg.timestamp:
                try:
                    timestamps.append(datetime.fromisoformat(msg.timestamp))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).days + 1

    def _get_person_candidates(self, messages: List[NormalizedMessage]) -> List[str]:
        """Get most active senders as person candidates."""
        from collections import Counter
        counts = Counter(msg.sender for msg in messages)
        return [name for name, _ in counts.most_common(10)]


# ─── Telegram JSON Parser ─────────────────────────────────────────────────────

class TelegramJSONParser:
    """Parse Telegram Desktop JSON export (result.json).

    Telegram Desktop export format:
    {
        "name": "Chat Name",
        "type": "public_channel" | "personal_chat" | "private_group",
        "messages": [
            {
                "id": 123,
                "type": "message",
                "date": "2024-01-15T14:30:00",
                "from": "User Name",
                "from_id": "user123",
                "text": "Hello",
                "text_entities": [{"type": "plain", "text": "Hello"}],
                "reply_to_message_id": 120
            }
        ]
    }
    """

    def parse(self, file_path: str) -> ImportResult:
        """Parse a Telegram export JSON file."""
        path = Path(file_path)
        if not path.exists():
            return ImportResult(
                messages=[], platform=ChatPlatform.TELEGRAM,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"文件不存在: {file_path}"],
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return ImportResult(
                messages=[], platform=ChatPlatform.TELEGRAM,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"JSON解析失败: {e}"],
            )

        return self.parse_json(data)

    def parse_json(self, data: dict) -> ImportResult:
        """Parse Telegram export data."""
        messages = []
        errors = []
        raw_messages = data.get("messages", [])

        # Build reply index for reply_to resolution
        reply_index = {}
        for raw in raw_messages:
            msg_id = raw.get("id")
            text = self._extract_text(raw)
            if msg_id and text:
                reply_index[msg_id] = text[:100]  # first 100 chars for context

        for raw in raw_messages:
            try:
                msg = self._parse_message(raw, reply_index)
                if msg:
                    messages.append(msg)
            except Exception as e:
                errors.append(f"消息#{raw.get('id', '?')}解析失败: {e}")

        time_span = self._compute_time_span(messages)
        candidates = self._get_person_candidates(messages)

        return ImportResult(
            messages=messages,
            platform=ChatPlatform.TELEGRAM,
            person_candidates=candidates,
            time_span_days=time_span,
            total_raw_messages=len(raw_messages),
            parse_errors=errors,
            metadata={
                "chat_name": data.get("name", ""),
                "chat_type": data.get("type", ""),
            },
        )

    def _parse_message(self, raw: dict, reply_index: dict) -> Optional[NormalizedMessage]:
        """Parse a single Telegram message."""
        msg_type = raw.get("type", "")
        if msg_type != "message":
            return None  # skip service messages

        sender = raw.get("from", "") or raw.get("actor", "") or "unknown"
        text = self._extract_text(raw)
        if not text:
            return None

        timestamp = raw.get("date", "")

        # Detect media type
        detected_type = "text"
        if raw.get("photo"):
            detected_type = "image"
        elif raw.get("media_type") == "video_message":
            detected_type = "video"
        elif raw.get("media_type") == "voice_message":
            detected_type = "audio"
        elif raw.get("media_type") == "sticker":
            detected_type = "sticker"
        elif raw.get("media_type") == "animation":
            detected_type = "sticker"
        elif raw.get("file"):
            detected_type = "file"

        # Reply chain
        reply_to = ""
        reply_id = raw.get("reply_to_message_id")
        if reply_id and reply_id in reply_index:
            reply_to = reply_index[reply_id]

        return NormalizedMessage(
            sender=sender,
            text=text,
            timestamp=timestamp,
            msg_type=detected_type,
            platform="telegram",
            reply_to=reply_to,
            metadata={"telegram_id": raw.get("id")},
        )

    def _extract_text(self, raw: dict) -> str:
        """Extract text from Telegram message (handles text_entities format)."""
        text = raw.get("text", "")
        if isinstance(text, list):
            # Telegram uses array format with entities
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            text = "".join(parts)
        return text.strip()

    def _compute_time_span(self, messages: List[NormalizedMessage]) -> int:
        timestamps = []
        for msg in messages:
            if msg.timestamp:
                try:
                    timestamps.append(datetime.fromisoformat(msg.timestamp.replace("Z", "+00:00")))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).days + 1

    def _get_person_candidates(self, messages: List[NormalizedMessage]) -> List[str]:
        from collections import Counter
        counts = Counter(msg.sender for msg in messages)
        return [name for name, _ in counts.most_common(10)]


# ─── Feishu/Lark JSON Parser ──────────────────────────────────────────────────

class FeishuJSONParser:
    """Parse Feishu/Lark API response format.

    Feishu message list format (from IM API):
    {
        "items": [
            {
                "message_id": "om_xxx",
                "msg_type": "text",
                "create_time": "1716000000000",  # Unix timestamp in ms
                "sender": {
                    "sender_type": "user",
                    "sender_id": {"open_id": "ou_xxx", "union_id": "on_xxx"}
                },
                "body": {
                    "content": "{\"text\":\"Hello\"}"
                },
                "mentions": [{"key": "@_user_1", "name": "User Name"}]
            }
        ]
    }

    Also handles simplified format:
    [
        {"sender": "name", "text": "message", "timestamp": "2024-01-15T14:30:00"}
    ]
    """

    def parse(self, file_path: str) -> ImportResult:
        """Parse a Feishu JSON export file."""
        path = Path(file_path)
        if not path.exists():
            return ImportResult(
                messages=[], platform=ChatPlatform.FEISHU,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"文件不存在: {file_path}"],
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return ImportResult(
                messages=[], platform=ChatPlatform.FEISHU,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"JSON解析失败: {e}"],
            )

        return self.parse_json(data)

    def parse_json(self, data: Any) -> ImportResult:
        """Parse Feishu data (API response or simplified format)."""
        messages = []
        errors = []

        # Detect format
        if isinstance(data, list):
            # Simplified format: direct list of messages
            raw_messages = data
        elif isinstance(data, dict):
            # API response format
            raw_messages = data.get("items", data.get("messages", data.get("data", {}).get("items", [])))
            if isinstance(raw_messages, dict):
                raw_messages = raw_messages.get("items", [])
        else:
            return ImportResult(
                messages=[], platform=ChatPlatform.FEISHU,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=["不支持的数据格式"],
            )

        for i, raw in enumerate(raw_messages):
            try:
                msg = self._parse_message(raw)
                if msg:
                    messages.append(msg)
            except Exception as e:
                errors.append(f"消息#{i}解析失败: {e}")

        time_span = self._compute_time_span(messages)
        candidates = self._get_person_candidates(messages)

        return ImportResult(
            messages=messages,
            platform=ChatPlatform.FEISHU,
            person_candidates=candidates,
            time_span_days=time_span,
            total_raw_messages=len(raw_messages),
            parse_errors=errors,
        )

    def _parse_message(self, raw: dict) -> Optional[NormalizedMessage]:
        """Parse a single Feishu message."""
        # Check for simplified format first
        if "sender" in raw and "text" in raw and isinstance(raw.get("sender"), str):
            return NormalizedMessage(
                sender=raw["sender"],
                text=raw["text"],
                timestamp=raw.get("timestamp", ""),
                msg_type=raw.get("msg_type", "text"),
                platform="feishu",
            )

        # Full API format
        msg_type = raw.get("msg_type", "text")
        create_time = raw.get("create_time", "")

        # Parse timestamp (Unix ms → ISO)
        timestamp = ""
        if create_time:
            try:
                ts_ms = int(create_time)
                dt = datetime.fromtimestamp(ts_ms / 1000)
                timestamp = dt.isoformat()
            except (ValueError, TypeError, OSError):
                timestamp = str(create_time)

        # Parse body content
        body = raw.get("body", {})
        content_str = body.get("content", "{}")
        if isinstance(content_str, str):
            try:
                content = json.loads(content_str)
            except json.JSONDecodeError:
                content = {"text": content_str}
        else:
            content = content_str

        # Extract text based on message type
        text = ""
        if msg_type == "text":
            text = content.get("text", "")
        elif msg_type == "post":
            # Rich text: extract all text from nested structure
            text = self._extract_post_text(content)
            msg_type = "post"
        elif msg_type == "interactive":
            text = content.get("header", {}).get("title", {}).get("content", "")
            if not text:
                text = json.dumps(content, ensure_ascii=False)[:200]
        elif msg_type == "image":
            text = "[图片]"
        elif msg_type == "audio":
            text = "[语音]"
        elif msg_type == "video":
            text = "[视频]"
        elif msg_type == "file":
            text = f"[文件] {content.get('file_name', '')}"
        elif msg_type == "merge_forward":
            text = "[合并转发]"
        else:
            text = json.dumps(content, ensure_ascii=False)[:200] if content else ""

        if not text:
            return None

        # Resolve sender name from mentions or sender_id
        sender = self._resolve_sender(raw)

        # Resolve mentions in text
        mentions = raw.get("mentions", [])
        for mention in mentions:
            key = mention.get("key", "")
            name = mention.get("name", "")
            if key and name:
                text = text.replace(key, f"@{name}")

        return NormalizedMessage(
            sender=sender,
            text=text.strip(),
            timestamp=timestamp,
            msg_type=msg_type,
            platform="feishu",
            metadata={
                "message_id": raw.get("message_id", ""),
                "sender_id": raw.get("sender", {}).get("sender_id", {}),
            },
        )

    def _extract_post_text(self, content: dict) -> str:
        """Extract plain text from Feishu rich text (post) message."""
        parts = []
        # Post format: {"zh_cn": {"title": "...", "content": [[{"tag":"text","text":"..."}]]}}
        # OR: {"title": "...", "content": [[...]]}
        post = content
        for lang_key in ["zh_cn", "en_us", "ja_jp"]:
            if lang_key in content:
                post = content[lang_key]
                break

        title = post.get("title", "")
        if title:
            parts.append(title)

        for paragraph in post.get("content", []):
            for element in paragraph:
                tag = element.get("tag", "")
                if tag == "text":
                    parts.append(element.get("text", ""))
                elif tag == "a":
                    parts.append(element.get("text", "") + f" ({element.get('href', '')})")
                elif tag == "at":
                    parts.append(f"@{element.get('user_name', 'user')}")
                elif tag == "img":
                    parts.append("[图片]")

        return "\n".join(parts)

    def _resolve_sender(self, raw: dict) -> str:
        """Resolve sender name from message metadata."""
        # If sender field is already a name string
        sender = raw.get("sender", {})
        if isinstance(sender, str):
            return sender

        # Try sender_name field
        if "sender_name" in raw:
            return raw["sender_name"]

        # From mentions (self-reference)
        sender_id = sender.get("sender_id", {}) if isinstance(sender, dict) else {}

        # Use open_id as fallback identifier
        open_id = sender_id.get("open_id", "")
        if open_id:
            return open_id

        return "unknown"

    def _compute_time_span(self, messages: List[NormalizedMessage]) -> int:
        timestamps = []
        for msg in messages:
            if msg.timestamp:
                try:
                    timestamps.append(datetime.fromisoformat(msg.timestamp))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).days + 1

    def _get_person_candidates(self, messages: List[NormalizedMessage]) -> List[str]:
        from collections import Counter
        counts = Counter(msg.sender for msg in messages)
        return [name for name, _ in counts.most_common(10)]


# ─── Generic CSV Parser ───────────────────────────────────────────────────────

class GenericCSVParser:
    """Parse generic CSV chat exports.

    Expected columns: sender, text, timestamp (flexible header matching).
    Also handles: from, message, time, content, name, msg, date.
    """

    SENDER_ALIASES = {"sender", "from", "name", "user", "发送者", "发送人", "昵称", "用户"}
    TEXT_ALIASES = {"text", "message", "content", "msg", "消息", "内容", "消息内容"}
    TIME_ALIASES = {"timestamp", "time", "date", "datetime", "时间", "日期", "发送时间"}

    def parse(self, file_path: str) -> ImportResult:
        """Parse a CSV chat export file."""
        path = Path(file_path)
        if not path.exists():
            return ImportResult(
                messages=[], platform=ChatPlatform.GENERIC_CSV,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"文件不存在: {file_path}"],
            )

        messages = []
        errors = []
        raw_count = 0

        try:
            # Try different encodings
            for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]:
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return ImportResult(
                    messages=[], platform=ChatPlatform.GENERIC_CSV,
                    person_candidates=[], time_span_days=0,
                    total_raw_messages=0,
                    parse_errors=["无法识别文件编码"],
                )

            # Detect delimiter
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(content[:4096])
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(content.splitlines(), dialect=dialect)
            if not reader.fieldnames:
                return ImportResult(
                    messages=[], platform=ChatPlatform.GENERIC_CSV,
                    person_candidates=[], time_span_days=0,
                    total_raw_messages=0,
                    parse_errors=["CSV无表头"],
                )

            # Map column names
            col_map = self._map_columns(reader.fieldnames)

            for row in reader:
                raw_count += 1
                try:
                    sender = row.get(col_map["sender"], "unknown").strip()
                    text = row.get(col_map["text"], "").strip()
                    timestamp = row.get(col_map["time"], "").strip()

                    if not text:
                        continue

                    messages.append(NormalizedMessage(
                        sender=sender,
                        text=text,
                        timestamp=self._normalize_timestamp(timestamp),
                        msg_type="text",
                        platform="csv",
                    ))
                except Exception as e:
                    errors.append(f"行#{raw_count}: {e}")

        except Exception as e:
            return ImportResult(
                messages=[], platform=ChatPlatform.GENERIC_CSV,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"CSV解析失败: {e}"],
            )

        time_span = self._compute_time_span(messages)
        candidates = self._get_person_candidates(messages)

        return ImportResult(
            messages=messages,
            platform=ChatPlatform.GENERIC_CSV,
            person_candidates=candidates,
            time_span_days=time_span,
            total_raw_messages=raw_count,
            parse_errors=errors,
        )

    def _map_columns(self, fieldnames) -> Dict[str, str]:
        """Map CSV columns to standard fields."""
        mapping = {"sender": "", "text": "", "time": ""}

        for field in fieldnames:
            lower = field.lower().strip()
            if not mapping["sender"] and lower in self.SENDER_ALIASES:
                mapping["sender"] = field
            elif not mapping["text"] and lower in self.TEXT_ALIASES:
                mapping["text"] = field
            elif not mapping["time"] and lower in self.TIME_ALIASES:
                mapping["time"] = field

        # Fallback: positional
        if not mapping["sender"] and fieldnames:
            mapping["sender"] = fieldnames[0]
        if not mapping["text"] and len(fieldnames) > 1:
            mapping["text"] = fieldnames[1]
        if not mapping["time"] and len(fieldnames) > 2:
            mapping["time"] = fieldnames[2]

        return mapping

    def _normalize_timestamp(self, ts: str) -> str:
        """Normalize timestamp to ISO 8601."""
        if not ts:
            return ""
        for fmt in [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        ]:
            try:
                return datetime.strptime(ts.strip(), fmt).isoformat()
            except ValueError:
                continue
        return ts

    def _compute_time_span(self, messages: List[NormalizedMessage]) -> int:
        timestamps = []
        for msg in messages:
            if msg.timestamp:
                try:
                    timestamps.append(datetime.fromisoformat(msg.timestamp))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).days + 1

    def _get_person_candidates(self, messages: List[NormalizedMessage]) -> List[str]:
        from collections import Counter
        counts = Counter(msg.sender for msg in messages)
        return [name for name, _ in counts.most_common(10)]


# ─── Auto-Detection & Unified Import ──────────────────────────────────────────

class ChatImporter:
    """Unified chat importer with auto-detection.

    Usage:
        importer = ChatImporter()
        result = importer.import_file("chat.html")
        result = importer.import_file("result.json")
        result = importer.import_file("messages.csv")

        # Or force platform
        result = importer.import_file("data.json", platform=ChatPlatform.FEISHU)

        # Get pipeline-ready data
        messages = result.to_pipeline_messages()
    """

    def __init__(self):
        self._parsers = {
            ChatPlatform.WECHAT: WeChatHTMLParser(),
            ChatPlatform.TELEGRAM: TelegramJSONParser(),
            ChatPlatform.FEISHU: FeishuJSONParser(),
            ChatPlatform.GENERIC_CSV: GenericCSVParser(),
        }

    def import_file(self, file_path: str, platform: Optional[ChatPlatform] = None) -> ImportResult:
        """Import a chat export file with auto-detection or forced platform."""
        path = Path(file_path)

        if not path.exists():
            return ImportResult(
                messages=[], platform=ChatPlatform.GENERIC_JSON,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"文件不存在: {file_path}"],
            )

        if platform:
            return self._parsers[platform].parse(file_path)

        # Auto-detect based on extension and content
        detected = self._detect_platform(file_path)
        logger.info("Auto-detected platform: %s for file: %s", detected.value, file_path)
        return self._parsers[detected].parse(file_path)

    def import_json_data(self, data: Any, platform: ChatPlatform = ChatPlatform.FEISHU) -> ImportResult:
        """Import from in-memory JSON data."""
        if platform == ChatPlatform.TELEGRAM:
            return TelegramJSONParser().parse_json(data)
        elif platform == ChatPlatform.FEISHU:
            return FeishuJSONParser().parse_json(data)
        else:
            return ImportResult(
                messages=[], platform=platform,
                person_candidates=[], time_span_days=0,
                total_raw_messages=0,
                parse_errors=[f"不支持从内存数据导入 {platform.value} 格式"],
            )

    def _detect_platform(self, file_path: str) -> ChatPlatform:
        """Detect chat platform from file content."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        # HTML files → WeChat
        if suffix in (".html", ".htm"):
            return ChatPlatform.WECHAT

        # CSV files
        if suffix == ".csv":
            return ChatPlatform.GENERIC_CSV

        # JSON files → inspect content
        if suffix == ".json":
            try:
                content = path.read_text(encoding="utf-8")[:8192]
                data = json.loads(path.read_text(encoding="utf-8"))

                # Telegram: has "messages" array with "from" and "text_entities"
                if isinstance(data, dict):
                    if "messages" in data and isinstance(data["messages"], list):
                        first = data["messages"][0] if data["messages"] else {}
                        if "text_entities" in first or "from_id" in first:
                            return ChatPlatform.TELEGRAM
                        if "msg_type" in first and "body" in first:
                            return ChatPlatform.FEISHU
                        if "sender" in first and "text" in first:
                            return ChatPlatform.FEISHU  # simplified feishu format

                    if "items" in data:
                        return ChatPlatform.FEISHU

                # Array of messages
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        if "text_entities" in first:
                            return ChatPlatform.TELEGRAM
                        if "sender" in first and "text" in first:
                            return ChatPlatform.FEISHU

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Default JSON → try Feishu (most common in this ecosystem)
            return ChatPlatform.FEISHU

        return ChatPlatform.GENERIC_JSON

    @staticmethod
    def to_pipeline_messages(result: ImportResult, person_name: str = "") -> List[dict]:
        """Convert ImportResult to pipeline-ready message format.

        Returns list of dicts compatible with conversation_analyzer.py and
        workflow_engine.py.
        """
        return [msg.to_dict() for msg in result.messages]

    @staticmethod
    def get_person_stats(result: ImportResult) -> dict:
        """Get statistics about senders for person selection."""
        from collections import Counter

        sender_counts = Counter(msg.sender for msg in result.messages)
        total = len(result.messages)

        stats = {}
        for sender, count in sender_counts.most_common(20):
            # Calculate average message length for this sender
            sender_msgs = [msg for msg in result.messages if msg.sender == sender]
            avg_len = sum(len(msg.text) for msg in sender_msgs) / max(len(sender_msgs), 1)

            # Calculate active days
            dates = set()
            for msg in sender_msgs:
                if msg.timestamp:
                    try:
                        dates.add(msg.timestamp[:10])
                    except (ValueError, IndexError):
                        pass

            stats[sender] = {
                "message_count": count,
                "percentage": round(count / total * 100, 1),
                "avg_message_length": round(avg_len, 1),
                "active_days": len(dates),
            }

        return stats


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if len(sys.argv) < 2:
        print("用法: python chat_importer.py <聊天记录文件路径> [平台]")
        print()
        print("支持的文件格式:")
        print("  .html/.htm  — WeChat HTML 导出")
        print("  .json       — Telegram JSON 导出 / Feishu JSON 导出 (自动检测)")
        print("  .csv        — 通用 CSV 格式 (sender,text,timestamp)")
        print()
        print("可选平台参数: wechat, telegram, feishu, csv")
        sys.exit(1)

    file_path = sys.argv[1]
    platform = None
    if len(sys.argv) > 2:
        try:
            platform = ChatPlatform(sys.argv[2])
        except ValueError:
            print(f"未知平台: {sys.argv[2]}")
            print(f"可选: {', '.join(p.value for p in ChatPlatform)}")
            sys.exit(1)

    importer = ChatImporter()
    result = importer.import_file(file_path, platform=platform)

    print("=== 导入结果 ===")
    print(result.summary())

    if result.parse_errors:
        print(f"\n=== 解析错误 ({len(result.parse_errors)}) ===")
        for err in result.parse_errors[:5]:
            print(f"  ⚠ {err}")

    if result.success:
        print(f"\n=== 人物统计 ===")
        stats = importer.get_person_stats(result)
        for name, info in list(stats.items())[:5]:
            print(f"  {name}: {info['message_count']}条 ({info['percentage']}%), "
                  f"均长{info['avg_message_length']}字, {info['active_days']}天活跃")

        print(f"\n=== 示例消息 (前5条) ===")
        for msg in result.messages[:5]:
            print(f"  [{msg.sender}] {msg.text[:50]}{'...' if len(msg.text) > 50 else ''}")
