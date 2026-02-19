"""
企业微信消息加解密模块。
参考官方文档: https://developer.work.weixin.qq.com/document/path/90968

消息流程:
  收消息: XML(密文) -> 验签 -> AES解密 -> 明文XML
  发消息: 明文XML -> AES加密 -> 签名 -> XML(密文)
"""
import base64
import hashlib
import random
import socket
import string
import struct
import time
import xml.etree.ElementTree as ET

from Crypto.Cipher import AES


class WeComCrypto:
    """企业微信消息加解密"""

    BLOCK_SIZE = 32

    def __init__(self, token: str, encoding_aes_key: str, corp_id: str):
        self.token = token
        self.corp_id = corp_id
        # EncodingAESKey 是 Base64 编码的 AES 密钥（43字符 + "=" -> 32字节）
        self.aes_key = base64.b64decode(encoding_aes_key + "=")

    @staticmethod
    def _sha1_sign(*args: str) -> str:
        """SHA1 签名：将参数排序后拼接，取 SHA1"""
        sorted_str = "".join(sorted(args))
        return hashlib.sha1(sorted_str.encode("utf-8")).hexdigest()

    def verify_signature(
        self, msg_signature: str, timestamp: str, nonce: str, echostr: str = ""
    ) -> bool:
        """验证签名（用于 URL 验证和消息验签）"""
        expected = self._sha1_sign(self.token, timestamp, nonce, echostr)
        return expected == msg_signature

    # ─── PKCS#7 Padding ────────────────────────────────────────

    def _pkcs7_pad(self, data: bytes) -> bytes:
        pad_len = self.BLOCK_SIZE - (len(data) % self.BLOCK_SIZE)
        return data + bytes([pad_len] * pad_len)

    @staticmethod
    def _pkcs7_unpad(data: bytes) -> bytes:
        pad_len = data[-1]
        return data[:-pad_len]

    # ─── 解密 ──────────────────────────────────────────────────

    def decrypt(self, encrypted_text: str) -> str:
        """
        解密企业微信密文。
        encrypted_text: Base64 编码的密文
        返回: 明文消息内容（XML 字符串）
        """
        cipher_bytes = base64.b64decode(encrypted_text)
        iv = self.aes_key[:16]
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(cipher_bytes)
        plaintext = self._pkcs7_unpad(decrypted)

        # 格式: 16字节随机串 + 4字节消息长度(网络字节序) + 消息内容 + CorpID
        msg_len = struct.unpack("!I", plaintext[16:20])[0]
        msg = plaintext[20 : 20 + msg_len].decode("utf-8")
        from_corp_id = plaintext[20 + msg_len :].decode("utf-8")

        if from_corp_id != self.corp_id:
            raise ValueError(
                f"CorpID 不匹配: 期望 {self.corp_id}, 收到 {from_corp_id}"
            )
        return msg

    # ─── 加密 ──────────────────────────────────────────────────

    def encrypt(self, reply_msg: str) -> str:
        """
        加密回复消息。
        reply_msg: 明文回复内容
        返回: Base64 编码的密文
        """
        random_prefix = "".join(
            random.choices(string.ascii_letters + string.digits, k=16)
        ).encode("utf-8")
        msg_bytes = reply_msg.encode("utf-8")
        corp_id_bytes = self.corp_id.encode("utf-8")

        plaintext = (
            random_prefix
            + struct.pack("!I", len(msg_bytes))
            + msg_bytes
            + corp_id_bytes
        )
        padded = self._pkcs7_pad(plaintext)

        iv = self.aes_key[:16]
        cipher = AES.new(self.aes_key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(padded)
        return base64.b64encode(encrypted).decode("utf-8")

    # ─── 解析收到的 XML ────────────────────────────────────────

    def decrypt_message(
        self,
        post_body: str,
        msg_signature: str,
        timestamp: str,
        nonce: str,
    ) -> str:
        """
        完整的消息解密流程:
        1. 从 XML 中提取 Encrypt 字段
        2. 验证签名
        3. 解密
        返回明文 XML
        """
        root = ET.fromstring(post_body)
        encrypt_node = root.find("Encrypt")
        if encrypt_node is None or encrypt_node.text is None:
            raise ValueError("XML 中未找到 Encrypt 字段")

        encrypted_text = encrypt_node.text

        # 验证签名
        expected_sign = self._sha1_sign(
            self.token, timestamp, nonce, encrypted_text
        )
        if expected_sign != msg_signature:
            raise ValueError("消息签名验证失败")

        return self.decrypt(encrypted_text)

    # ─── 构造加密回复 XML ──────────────────────────────────────

    def encrypt_message(self, reply_msg: str, nonce: str) -> str:
        """
        加密回复消息并封装为 XML。
        reply_msg: 明文回复内容（XML 格式的消息体）
        返回: 加密后的完整 XML 响应
        """
        encrypted_text = self.encrypt(reply_msg)
        timestamp = str(int(time.time()))
        signature = self._sha1_sign(
            self.token, timestamp, nonce, encrypted_text
        )

        return (
            "<xml>"
            f"<Encrypt><![CDATA[{encrypted_text}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            "</xml>"
        )


def parse_text_message(xml_str: str) -> dict:
    """
    从明文 XML 中解析文本消息。
    返回 dict: {to_user, from_user, create_time, msg_type, content, msg_id, agent_id}
    """
    root = ET.fromstring(xml_str)
    return {
        "to_user": (root.findtext("ToUserName") or "").strip(),
        "from_user": (root.findtext("FromUserName") or "").strip(),
        "create_time": (root.findtext("CreateTime") or "").strip(),
        "msg_type": (root.findtext("MsgType") or "").strip(),
        "content": (root.findtext("Content") or "").strip(),
        "msg_id": (root.findtext("MsgId") or "").strip(),
        "agent_id": (root.findtext("AgentID") or "").strip(),
    }


def build_text_reply_xml(
    from_user: str, to_user: str, content: str
) -> str:
    """构造被动回复的明文 XML（文本消息）"""
    timestamp = str(int(time.time()))
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{timestamp}</CreateTime>"
        f"<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )
