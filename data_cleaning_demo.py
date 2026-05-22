"""
═══════════════════════════════════════════════════════════════════════════════
            RAG 数据清洗与预处理 —— 完整实战代码
═══════════════════════════════════════════════════════════════════════════════

依赖：
  pip install pandas opencv-python numpy Pillow

背景：
  在 RAG 系统中，数据清洗是"检索→增强→生成"之前的第一道关卡。
  原始文档通常包含大量噪声，直接向量化入库会导致：
    · 检索精度下降（噪声词污染向量空间）
    · 切分边界不准（多余的换行/空格破坏语义完整性）
    · LLM 理解困难（格式混乱、无效字符混入）

  本文件的四个部分：
    1. 文本清洗 ——  去除 HTML 标签、特殊符号、合并空白符
    2. 图像预处理 —— 二值化、降噪、倾斜矫正（提高 OCR 准确率）
    3. 代码清洗 ——  去除注释、统一缩进、去除空行
    4. 混合文档管线 —— 自动识别文档类型，分发到对应清洗器
═══════════════════════════════════════════════════════════════════════════════
"""
import re
import os
import sys
from typing import Callable, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


# ============================================================================
# 第一部分：文本清洗 —— pandas 实战
# ============================================================================

def demo_text_cleaning():
    """
    文本清洗的目标：
      1. 去除无关内容（HTML标签、特殊符号、控制字符）
      2. 统一格式（合并多余空格/换行/制表符，让文本变成"干净段落"）
      3. 提高切分准确性（整洁的文本 → Tokenizer 能正确数 token → chunk 边界准确）
      4. 提高 LLM 阅读准确度（LLM 不会被垃圾字符干扰）

    本例用 pandas 的原因：
      - 企业 RAG 中，数据往往以 CSV/Excel 形式存在（FAQ列表、产品库、知识条目）
      - pandas 可以批量处理多行文本，而非逐行循环
      - 清洗结果可以直接导出为干净的 CSV 或喂给后续管线
    """
    import pandas as pd

    print("=" * 70)
    print("  第一部分：文本清洗")
    print("=" * 70)

    # ── 原始数据（模拟从企业 FAQ 系统导出的脏数据）────────────
    raw_data = [
        # 第 1 条：混入 HTML 标签 + 多余空格
        "<div class='faq'><b>Q:</b> 请问  事假 需要   提前多久  申请？</div>",

        # 第 2 条：混入特殊字符 + 多余换行
        "A: 事假需提前\u00a0\u00a01个工作日\u00a0申请。\n\n\n（紧急情况可事后补办）\r\n\t\t",

        # 第 3 条：混入 URL + 零宽字符 + 控制字符
        "详见员工手册：https://wiki.internal.com/policy?id=123 \x00\x01 或联系HR。",

        # 第 4 条：纯垃圾（空白行、只包含符号的无效内容）
        "\n\n  ---  \n\n  ***  \n",

        # 第 5 条：正常文本，但有全角半角混用和多余标点
        "年假天数：入职１－５年享有５天年假，，，入职５－１０年享有１０天年假。。。",
    ]

    df = pd.DataFrame({"raw_text": raw_data, "source": "FAQ系统"})

    print("\n  [清洗前] 原始数据：")
    print(df.to_string(max_colwidth=60))

    # ── 清洗函数（用 .apply() 的向量化操作）─────────────────

    def clean_text(text: str) -> str:
        """
        对一个文本字符串执行完整的清洗流水线。

        每一步的作用都有注释说明，你可以根据实际文档特点
        增删步骤。
        """
        # Step 1: 移除 HTML / XML 标签
        #   正则 <[^>]*> 匹配从 < 到 > 的任意内容（非贪婪）
        text = re.sub(r'<[^>]*>', '', text)

        # Step 2: 移除控制字符
        #   \x00-\x08, \x0b-\x0c, \x0e-\x1f 是不可见控制字符
        #   它们来自二进制文件混入、OCR 错误等，对 RAG 无意义
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Step 3: 移除零宽字符
        #   \u200b（零宽空格）、\u200c（零宽非连接符）等
        #   OCR 或复制粘贴时常引入，肉眼不可见但影响 tokenizer
        text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', text)

        # Step 4: 将不间断空格 (Non-breaking space, \u00a0) 替换为普通空格
        #   \u00a0 看起来像空格，但不是 ASCII 空格，
        #   可能导致分词器把它当成一个独立 token，破坏语义
        text = text.replace('\u00a0', ' ')
        #   全角空格也一并处理
        text = text.replace('\u3000', ' ')

        # Step 5: 移除 URL（可选，取决于你的场景）
        #   如果需要保留 URL 中的信息（比如问答对中 "详见 https://..."），
        #   则跳过这一步。反之如果 URL 是噪声，则移除。
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Step 6: 将多个连续的换行/回车符合并为一个 \\n
        text = re.sub(r'[\r\n]+', '\n', text)

        # Step 7: 将多个连续空格/制表符合并为一个空格
        text = re.sub(r'[ \t]+', ' ', text)

        # Step 8: 将 \\n 前后多余的空格去除
        text = re.sub(r' *\n *', '\n', text)

        # Step 9: 全角数字和字母转半角（中文文档常见问题）
        #   全角: ０１２３...ａｂｃ...ＡＢＣ
        #   半角: 0123...abc...ABC
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
            '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ))

        # Step 10: 合并连续重复的标点（去抖）
        text = re.sub(r'，{2,}', '，', text)  # 多个逗号 → 1个
        text = re.sub(r'。{2,}', '。', text)  # 多个句号 → 1个
        text = re.sub(r'、{2,}', '、', text)  # 多个顿号 → 1个

        # Step 11: 去除首尾空白，去除纯空白行
        text = text.strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        text = '\n'.join(lines)

        return text

    # ── 批量清洗 ──────────────────────────────────────────────
    # pandas 的 .apply() 是向量化操作，比 for 循环遍历每行更高效
    df["cleaned_text"] = df["raw_text"].apply(clean_text)

    # ── 过滤无效行 ───────────────────────────────────────────
    # 清洗后如果文本太少（比如之前是纯垃圾），直接丢弃
    df = df[df["cleaned_text"].str.len() >= 5]  # 小于 5 个字符视为无效

    print("\n  [清洗后] 清洗结果：")
    # 把清洗前后的文本都显示出来，方便对比
    for i, row in df.iterrows():
        raw = row["raw_text"][:80].replace("\n", "\\n")
        clean = row["cleaned_text"][:80].replace("\n", "\\n")
        print(f"  [{i}] 清洗前: {raw}...")
        print(f"      清洗后: {clean}...")
        print()

    return df


# ============================================================================
# 第二部分：图像文档预处理 —— cv2 实战
# ============================================================================

def demo_image_cleaning():
    """
    图像预处理的目标：
      在 OCR 之前，对扫描件/拍照文档做图像增强，提高 OCR 识别准确率。

    三个核心操作：

      1. 二值化 (Binarization)
         将彩色/灰度图像转为纯黑白（0或255）。
         为什么需要：OCR 引擎需要清晰的文字-背景对比。
                    灰度的过渡像素会造成误识别。

         ┌──────────────────┐      ┌──────────────────┐
         │  䵸灰色文字       │  →   │  ■ 纯黑文字       │
         │  䶅浅灰背景       │      │  □ 纯白背景       │
         └──────────────────┘      └──────────────────┘

      2. 降噪 (Denoising)
         去除扫描件/照片上的噪点（灰尘颗粒、传感器噪点、复印斑点）。
         为什么需要：噪点会被 OCR 误识别为标点符号或文字碎片。
                     "事假.需提前" → 因为一个噪点被识别为 "." 导致断词

      3. 倾斜矫正 (Deskewing)
         检测文档在扫描/拍照时的倾斜角度，旋转回水平。
         为什么需要：倾斜的文字在字符合并时会错位，
                     Tesseract 对倾斜 > 5° 的文字识别率急剧下降。
    """
    import cv2
    import numpy as np

    print("=" * 70)
    print("  第二部分：图像文档预处理（OCR 前置增强）")
    print("=" * 70)

    # ── 创建模拟的"扫描文档"图片 ──────────────────────────
    # 生产环境你会从 PDF 中提取页图片，或直接读取相机拍摄的文档照
    # 这里用 numpy 构造一张模拟图来演示（实际使用请替换为 cv2.imread）

    # 一张 400×600 的白色背景
    img = np.ones((600, 400), dtype=np.uint8) * 240  # 浅灰背景

    # 画一些"文字"（用矩形模拟笔画，故意做成灰色 + 倾斜）
    cv2.putText(img, "Hello World", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 3)
    cv2.putText(img, "RAG Data Cleaning", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 3)
    cv2.putText(img, "OCR Preprocessing", (40, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 3)

    # 添加随机噪点（模拟扫描件上的灰尘/斑点）
    np.random.seed(42)
    noise_mask = np.random.random(img.shape) > 0.98   # 2% 的像素加噪
    img[noise_mask] = np.random.randint(0, 150, size=noise_mask.sum())

    print(f"\n  原始图像 shape: {img.shape} , dtype: {img.dtype}")
    print(f"  像素值范围: [{img.min()}, {img.max()}]")

    # ═════════════════════════════════════════════════════════
    # 操作 1: 二值化
    # ═════════════════════════════════════════════════════════
    print("\n  --- 操作 1: 二值化 ---")

    # 方法 A: 固定阈值（适合光照均匀的扫描件）
    #   cv2.threshold 的 THRESH_BINARY 模式：
    #     像素 > 127 → 255（白）
    #     像素 ≤ 127 → 0（黑）
    _, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 方法 B: 自适应阈值（适合光照不均匀的拍照文档）
    #   cv2.adaptiveThreshold：
    #     每个像素的阈值 = 周围 11×11 区域的均值 - 2
    #     这样即使左上角偏暗、右下角偏亮，也能正确二值化
    binary_adaptive = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 用高斯加权计算局部阈值
        cv2.THRESH_BINARY,
        11,   # 邻域大小（必须是奇数）
        2,    # 减去常数
    )

    print(f"  固定阈值 → 仅剩 {np.sum(binary_fixed == 0)} 个黑色像素 (文字)")
    print(f"  自适应阈值 → 仅剩 {np.sum(binary_adaptive == 0)} 个黑色像素 (文字)")
    print(f"  对比：原始灰度像素 {np.sum(img < 128)} 个暗像素")
    print(f"  说明：二值化后，每个像素只有 0 或 255 两个值")
    print(f"        OCR 引擎现在面对的是清晰的「黑字白底」，而非模糊的灰度过渡")

    # ═════════════════════════════════════════════════════════
    # 操作 2: 降噪
    # ═════════════════════════════════════════════════════════
    print("\n  --- 操作 2: 降噪 ---")

    # 方法 A: 中值滤波 (Median Blur)
    #   原理：每个像素替换为周围 3×3 区域的中位数
    #   效果：椒盐噪点（孤立的黑白点）被周围像素"吞没"
    #   优点：保留边缘（不会把文字边缘"模糊"掉）
    denoised_median = cv2.medianBlur(binary_fixed, 3)

    # 方法 B: 高斯滤波 (Gaussian Blur) + 重新二值化
    #   原理：用高斯核对图像做卷积，每个像素替换为邻域的加权平均
    #   效果：高频噪声被"平滑"掉
    #   缺点：文字边缘也会被轻微模糊
    blurred = cv2.GaussianBlur(binary_fixed, (3, 3), 0)
    _, denoised_gaussian = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 噪点计数对比
    def count_noise(pixels, region_size=2):
        """简单估计：统计孤立的黑色像素（周围全是白的那个黑点）"""
        kernel = np.ones((region_size, region_size), np.uint8)
        isolated = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
        return abs(int(np.sum(pixels) - np.sum(isolated)))

    noise_fixed = count_noise(binary_fixed)
    noise_median = count_noise(denoised_median)
    noise_gaussian = count_noise(denoised_gaussian)

    print(f"  原始二值化图的噪点数（估算）: {noise_fixed}")
    print(f"  中值滤波后的噪点数（估算）  : {noise_median}  << 减少了 {noise_fixed - noise_median}")
    print(f"  高斯滤波后的噪点数（估算）  : {noise_gaussian}")
    print(f"  说明：二值化后的小黑点被平滑掉，减少了 OCR 误识别的概率")

    # ═════════════════════════════════════════════════════════
    # 操作 3: 倾斜矫正
    # ═════════════════════════════════════════════════════════
    print("\n  --- 操作 3: 倾斜矫正 ---")

    def deskew_image(image, max_angle: float = 15.0):
        """
        检测文档倾斜角度并旋转矫正。

        原理：
          1. 用 Canny 边缘检测找出文档中的笔画边缘
          2. 用霍夫变换 (Hough Transform) 找出这些边缘对应的直线
          3. 统计这些直线的角度分布
          4. 取角度的中位数作为文档的整体倾斜角
          5. 用仿射变换旋转回水平

        参数:
          image: 灰度图（二值化后的效果更好）
          max_angle: 最大检测角度（度），超过此角度认为是噪声直线

        返回:
          旋转矫正后的图像
        """
        # Step 1: 边缘检测
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Step 2: 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return image

        # Step 3: 收集所有直线的角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # theta 是弧度，转为角度
            angle = np.rad2deg(theta)
            # 矫正到 [-45, 45] 范围（与水平线的偏差）
            if 45 < angle < 135:
                angle = angle - 90
            if -135 < angle < -45:
                angle = angle + 90
            if abs(angle) < max_angle:
                angles.append(angle)

        if not angles:
            return image

        # Step 4: 取中位数角度
        median_angle = np.median(angles)
        print(f"  检测到倾斜角度: {median_angle:.2f}°")

        # Step 5: 旋转矫正
        h, w = image.shape
        center = (w // 2, h // 2)
        # 获取旋转矩阵（绕中心点旋转 -angle 度）
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        # 仿射变换
        deskewed = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,  # 三次插值（高质量旋转）
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # 旋转后空白区域填白色
        )

        return deskewed

    # 对二值化后的图做倾斜矫正
    deskewed = deskew_image(binary_fixed, max_angle=20.0)

    print(f"  矫正前 shape: {binary_fixed.shape}")
    print(f"  矫正后 shape: {deskewed.shape}")
    print(f"  说明：矫正后的图像文字为水平排列，OCR 引擎可以正确地")
    print(f"        对文字行做切割和识别，不会因为倾斜导致行间串字")

    # ── 完整预处理流水线 ─────────────────────────────────────
    print("\n  --- 图像预处理流水线 ---")
    print("  输入： 原始灰度图 (含噪点、灰底、倾斜)")
    print("  Step 1: 二值化   → 纯黑白，消除灰度过渡")
    print("  Step 2: 降噪     → 去除椒盐噪点")
    print("  Step 3: 倾斜矫正 → 旋转回水平")
    print("  输出： 干净的二值化图，可直接送入 OCR 引擎")
    print("  预估 OCR 准确率提升：10%-30%（取决于原图质量）")


# ============================================================================
# 第三部分：代码块清洗
# ============================================================================

def demo_code_cleaning():
    """
    代码清洗的目标：
      1. 去除注释（对 RAG 来说通常是噪声）
      2. 统一缩进（混合 Tab/空格的缩进会被 Tokenizer 切成多余 token）
      3. 去除空行/纯符号行
      4. 保留代码的结构语义（函数边界、类边界）

    为什么需要清洗代码：
      开发者文档、技术博客、代码仓库中的代码块通常含有大量注释、
      不一致的缩进、空行等。这些噪声在向量检索时会稀释语义——
      一段 Python 代码和一个自然语言问题做相似度匹配时，
      注释中的"# TODO: fix this" 比函数签名 "def calculate_overtime_pay"
      对"加班费怎么算"这个问题的匹配度低得多。
    """
    import pandas as pd

    print("=" * 70)
    print("  第三部分：代码块清洗")
    print("=" * 70)

    # ── 原始代码块（模拟从技术文档/代码仓库中提取的）───────
    raw_code = r'''
# 这是一个计算加班费的函数
# TODO: 后续需要支持调休计算
def calculate_overtime(hours, rate, is_holiday=False):
    """计算加班费
    Args:
        hours: 加班小时数
        rate: 小时工资
        is_holiday: 是否为节假日加班
    Returns:
        float: 加班费金额
    """
    # 基础倍数
    multiplier = 1.5

	if is_holiday:		# 节假日加班按 3 倍
		multiplier = 3.0
    # elif is_weekend:    # 这段先注释掉，后面再加
    #     multiplier = 2.0

    # 计算最终金额
    total = hours * rate * multiplier  # total = 加班费

    print("计算完成")  # 调试用的，后面删掉

    return total


# 旧版本代码（已废弃，保留备用）
# def old_calculate(hours, rate):
#     return hours * rate * 1.5
'''

    df = pd.DataFrame({"raw_code": [raw_code], "source": "技术文档"})

    print("\n  [清洗前] 原始代码块：")
    # 只显示前 20 行
    for line in raw_code.split('\n')[:20]:
        print(f"    |{line}")

    # ── 清洗函数 ──────────────────────────────────────────

    def clean_code(code: str, language: str = "python") -> str:
        """
        清洗代码块，使其适合 RAG 检索和 LLM 阅读。

        每一步的具体作用：
        """
        # Step 1: 去除单行注释（以 # 开头的整行注释）
        #   这些行不包含可执行的逻辑，是纯粹的噪声
        code = re.sub(r'^\s*#.*$', '', code, flags=re.MULTILINE)

        # Step 2: 去除文档字符串 (docstring)
        #   Python 的 """...""" 文档字符串在 RAG 中通常不如函数签名有价值
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Step 3: 去除行尾注释（如 print("hello") # 这是注释）
        #   但要小心：不能把字符串中的 # 也去掉
        #   简易处理：替换 'xxx # comment' 模式
        code = re.sub(r'  # .*$', '', code, flags=re.MULTILINE)

        # Step 4: 统一缩进——将所有 Tab 替换为 4 个空格
        code = code.replace('\t', '    ')

        # Step 5: 去除空行（连续的空行→1个空行→去除）
        code = re.sub(r'\n\s*\n', '\n', code)

        # Step 6: 去除行尾空格
        code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)

        # Step 7: 折行处理——将过长的行折叠（可选）
        #   这里不强制实现，取决于你的场景

        # Step 8: 去除 PRINT 调试语句（可选，取决于你的场景）
        #   生产代码中残留的 print() 对 RAG 无意义
        code = re.sub(r'^\s*print\(.*\).*$', '', code, flags=re.MULTILINE)

        return code.strip()

    # ── 执行清洗 ──────────────────────────────────────────
    df["cleaned_code"] = df["raw_code"].apply(clean_code)

    print("\n  [清洗后] 清洗后的代码块：")
    for line in df["cleaned_code"].iloc[0].split('\n'):
        if line.strip():
            print(f"    |{line}")

    # ── 对比分析 ──────────────────────────────────────────
    raw_lines = len(raw_code.split('\n'))
    clean_lines = len(df["cleaned_code"].iloc[0].split('\n')) if df["cleaned_code"].iloc[0] else 0
    raw_chars = len(raw_code)
    clean_chars = len(df["cleaned_code"].iloc[0]) if df["cleaned_code"].iloc[0] else 0

    print(f"\n  清洗效果：")
    print(f"    行数: {raw_lines} → {clean_lines} (减少 {raw_lines - clean_lines} 行)")
    print(f"    字符: {raw_chars} → {clean_chars} (减少 {raw_chars - clean_chars} 字符)")
    print(f"  说明：去除了注释、文档字符串、打印语句和空行")
    print(f"        保留的是函数签名和核心逻辑——这些才是检索时需要的语义信息")
    print(f"        向量化时'def calculate_overtime' 的语义权重更高")


# ============================================================================
# 第四部分：混合文档清洗管线
# ============================================================================

def demo_mixed_pipeline():
    """
    混合文档清洗管线：自动识别文档类型，分发到对应的清洗器。

    在生产环境中，RAG 系统接收的文档可能是：
      - 纯文本/Markdown  → 需要文本清洗
      - 扫描件 PDF        → 需要图像预处理 + OCR
      - 代码块            → 需要代码清洗
      - HTML 网页         → 需要 HTML 解析 + 文本清洗

    这个管线的核心模式：
      1. 识别文档类型（基于扩展名、MIME type 或内容特征）
      2. 分发到对应的清洗器
      3. 统一输出为标准 Document
    """
    print("=" * 70)
    print("  第四部分：混合文档清洗管线")
    print("=" * 70)

    # ── 文档类型检测 ──────────────────────────────────────
    # 简单的基于扩展名的检测（生产环境中可扩展为基于 magic bytes / MIME）

    DOC_TYPE_MAP: Dict[str, str] = {
        ".md":   "text",
        ".txt":  "text",
        ".csv":  "tabular",
        ".html": "html",
        ".htm":  "html",
        ".py":   "code",
        ".js":   "code",
        ".java": "code",
        ".go":   "code",
        ".ts":   "code",
        ".pdf":  "pdf",
        ".png":  "image",
        ".jpg":  "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".docx": "office",
        ".pptx": "office",
        ".xlsx": "tabular",
    }

    def detect_doc_type(file_path: str) -> str:
        """根据文件扩展名识别文档类型"""
        _, ext = os.path.splitext(file_path)
        return DOC_TYPE_MAP.get(ext.lower(), "unknown")

    # ── 清洗器注册表 ──────────────────────────────────────
    # 每个文档类型对应一个清洗函数
    # 这是"策略模式"——新增清洗类型只需注册新函数

    def clean_text_doc(text: str) -> str:
        """文本类文档的清洗（复用第一部分的逻辑）"""
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u3000', ' ')
        return text.strip()

    def clean_html_doc(html: str) -> str:
        """HTML 文档的清洗：去除标签、脚本、样式 + 文本清洗"""
        # 去除 <script>...</script> 和 <style>...</style> 块
        html = re.sub(r'<script[^>]*>.*?</script>', '', html,
                      flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html,
                      flags=re.DOTALL | re.IGNORECASE)
        # 去除 HTML 注释
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        # 然后用文本清洗函数处理剩余
        return clean_text_doc(html)

    def clean_code_doc(code: str) -> str:
        """代码文档的清洗（复用第三部分的逻辑）"""
        code = re.sub(r'^\s*#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'  # .*$', '', code, flags=re.MULTILINE)
        code = code.replace('\t', '    ')
        code = re.sub(r'\n\s*\n', '\n', code)
        return code.strip()

    def clean_tabular_doc(text: str) -> str:
        """表格数据清洗：保留行列结构，去除噪声"""
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.replace('\u00a0', ' ')
        return text.strip()

    def clean_unknown_doc(text: str) -> str:
        """未知类型：做基本的文本清洗"""
        return clean_text_doc(text)

    # ── 清洗器注册表 ──────────────────────────────────────
    CLEANER_REGISTRY: Dict[str, Callable[[str], str]] = {
        "text":    clean_text_doc,
        "html":    clean_html_doc,
        "code":    clean_code_doc,
        "tabular": clean_tabular_doc,
        "office":  clean_text_doc,   # Office 文档用 Unstructured 解析后到这里
        "pdf":     clean_text_doc,   # PDF 用 Unstructured 解析后到这里
        "image":   clean_unknown_doc, # 图像需要 OCR，清洗逻辑不同（见第二部分）
        "unknown": clean_unknown_doc,
    }

    # ── 统一清洗入口 ──────────────────────────────────────

    def clean_document(content: str, doc_type: str) -> str:
        """
        根据文档类型自动选择清洗器。

        这是整个清洗管线的"分发器"——调用方只需提供内容和类型，
        不需要知道内部清洗逻辑的细节。
        """
        cleaner = CLEANER_REGISTRY.get(doc_type, clean_unknown_doc)
        return cleaner(content)

    # ── 演示：模拟处理不同类型的文档 ──────────────────────
    test_files = [
        ("FAQ手册.md", "# 常见问题\n\n\n## Q1: 请假\n  事假 需提前  <b>1个工作日</b> 申请。\n\n\n"),
        ("index.html", "<html><head><script>track()</script></head><body><h1>政策</h1><p>事假需提前1个工作日</p></body></html>"),
        ("计算.py", "def calc(x):\n\t# 这是注释\n\ty = x * 1.5  # 倍数\n\treturn y\n\n"),
        ("数据.csv", "姓名,  部门,  年假天数\n张三,  技术,  5\n李四,  市场,  10\n"),
        ("报告.docx", "  第一章  概述\n\n本公司成立于2018年。  \n\n\n\n  \n"),
    ]

    for filepath, content in test_files:
        doc_type = detect_doc_type(filepath)
        cleaned = clean_document(content, doc_type)
        print(f"\n  [{doc_type}] {filepath}")
        print(f"    清洗前: {repr(content[:60])}...")
        print(f"    清洗后: {repr(cleaned[:60])}...")

    # ── 管线总结 ──────────────────────────────────────────
    print("\n\n  --- 混合清洗管线架构 ---")
    print("""
    📄 原始文件
       │
       ▼
    ┌─────────────────┐
    │  detect_doc_type │  ← 基于扩展名/MIME识别类型
    │   返回: "text"   │
    │         "html"   │
    │         "code"   │
    │         "image"  │  ← 图像走第二部分（cv2预处理→OCR）
    │         ...      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  clean_document  │  ← 分发器：根据类型调用对应的清洗器
    │  (统一入口)       │
    └────────┬────────┘
             │
    ┌────────┼────────┬────────┬────────┐
    ▼        ▼        ▼        ▼        ▼
  text    html     code    tabular   image
  清洗器   清洗器    清洗器   清洗器    预处理
             │
             ▼
    ┌─────────────────┐
    │  干净的文本       │  ← 输出：统一的清洗后文本
    │   (喂给切分器)   │
    └─────────────────┘
    """)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  RAG 数据清洗与预处理 —— 完整演示")
    print("=" * 70)

    # 第一部分：文本清洗
    demo_text_cleaning()

    # 第二部分：图像预处理
    try:
        import cv2
        demo_image_cleaning()
    except ImportError:
        print("\n  [WARN] 未安装 opencv-python。跳过图像预处理演示。")
        print("    安装: pip install opencv-python")

    # 第三部分：代码清洗
    demo_code_cleaning()

    # 第四部分：混合管线
    demo_mixed_pipeline()

    print("\n" + "=" * 70)
    print("  所有清洗演示完成")
    print("=" * 70)
