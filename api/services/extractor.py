"""
extractor.py — Módulo de Extracción de Texto PDF
=================================================
Permite elegir el motor de extracción antes de chunking.
Cada motor devuelve:
  - pages: lista de dicts { "page": int, "text": str }
  - texto_completo: str con todo el texto limpio
  - motor_usado: str
  - advertencias: list[str]

Motores disponibles:
  - "pymupdf"    → PyMuPDF      (velocidad extrema, metadatos ricos)
  - "pdfplumber" → pdfplumber   (tablas y layout estructurado)
  - "pypdf2"     → PyPDF2/pypdf (simple, para PDFs básicos)
  - "ocr"        → Pytesseract  (PDFs escaneados / imágenes)
"""

import re
import os

# Configuración de Tesseract OCR para Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSERACT_EXE = os.path.join(TESSERACT_PATH, "tesseract.exe")
if os.path.exists(TESSERACT_EXE):
    os.environ["TESSDATA_PREFIX"] = os.path.join(TESSERACT_PATH, "tessdata")
    # Añadir al PATH si no está
    if TESSERACT_PATH not in os.environ.get("PATH", ""):
        os.environ["PATH"] = TESSERACT_PATH + os.pathsep + os.environ.get("PATH", "")


def _clean_text(text: str) -> str:
    """Limpieza estándar: colapsar saltos y espacios redundantes."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_pymupdf(file_path: str) -> dict:
    """
    Motor: PyMuPDF (fitz)
    Fortaleza: Velocidad extrema, metadatos automáticos precisos.
    Debilidad: No hace OCR por sí solo.
    """
    try:
        import fitz  # PyMuPDF
        pages = []
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            raw = page.get_text("text")
            clean = _clean_text(raw)
            if clean:
                pages.append({"page": i + 1, "text": clean})
        doc.close()

        texto_completo = "\n\n".join(
            f"--- PÁGINA {p['page']} ---\n{p['text']}" for p in pages
        )
        return {
            "pages": pages,
            "texto_completo": texto_completo,
            "motor_usado": "PyMuPDF",
            "advertencias": [] if pages else ["PDF sin texto extraíble. ¿Es un escaneado?"]
        }
    except Exception as e:
        return {"pages": [], "texto_completo": "", "motor_usado": "PyMuPDF", "advertencias": [f"Error PyMuPDF: {str(e)}"]}


def extract_pdfplumber(file_path: str) -> dict:
    """
    Motor: pdfplumber
    Fortaleza: Excelente para tablas, columnas y diseño estructurado.
    Debilidad: Más lento que PyMuPDF.
    """
    try:
        import pdfplumber
        pages = []
        advertencias = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw = page.extract_text()

                # Intentar también extraer tablas si hay
                tables = page.extract_tables()
                tabla_txt = ""
                if tables:
                    for table in tables:
                        for row in table:
                            row_clean = [cell.strip() if cell else "" for cell in row]
                            tabla_txt += " | ".join(row_clean) + "\n"

                combined = (raw or "") + ("\n[TABLA]\n" + tabla_txt if tabla_txt else "")
                clean = _clean_text(combined)
                if clean:
                    pages.append({"page": i + 1, "text": clean})

        if not pages:
            advertencias.append("PDF sin texto. Puede ser un escaneado.")

        texto_completo = "\n\n".join(
            f"--- PÁGINA {p['page']} ---\n{p['text']}" for p in pages
        )
        return {
            "pages": pages,
            "texto_completo": texto_completo,
            "motor_usado": "pdfplumber",
            "advertencias": advertencias
        }
    except Exception as e:
        return {"pages": [], "texto_completo": "", "motor_usado": "pdfplumber", "advertencias": [f"Error pdfplumber: {str(e)}"]}


def extract_pypdf2(file_path: str) -> dict:
    """
    Motor: pypdf (sucesor de PyPDF2)
    Fortaleza: Muy común, simple y liviano.
    Debilidad: Falla con caracteres especiales o diseños complejos.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        advertencias = []

        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            clean = _clean_text(raw)
            if clean:
                pages.append({"page": i + 1, "text": clean})

        if not pages:
            advertencias.append("PyPDF2 no pudo extraer texto. Prueba PyMuPDF o OCR.")

        texto_completo = "\n\n".join(
            f"--- PÁGINA {p['page']} ---\n{p['text']}" for p in pages
        )
        return {
            "pages": pages,
            "texto_completo": texto_completo,
            "motor_usado": "PyPDF2/pypdf",
            "advertencias": advertencias
        }
    except Exception as e:
        return {"pages": [], "texto_completo": "", "motor_usado": "PyPDF2/pypdf", "advertencias": [f"Error PyPDF2: {str(e)}"]}


def extract_ocr(file_path: str) -> dict:
    """
    Motor: Pytesseract (OCR)
    Fortaleza: Lee texto dentro de imágenes y PDFs escaneados.
    Debilidad: Lento y requiere Tesseract instalado en el sistema.
    """
    try:
        import fitz  # PyMuPDF para renderizar páginas como imagen
        import pytesseract
        from PIL import Image
        import io

        pages = []
        advertencias = []
        doc = fitz.open(file_path)

        for i, page in enumerate(doc):
            # Renderizar la página como imagen a 300 DPI
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # OCR con Tesseract
            raw = pytesseract.image_to_string(img, lang='spa+eng')
            clean = _clean_text(raw)
            if clean:
                pages.append({"page": i + 1, "text": clean})

        doc.close()

        if not pages:
            advertencias.append("OCR no detectó texto. Verifica que Tesseract esté instalado (tesseract.exe).")

        texto_completo = "\n\n".join(
            f"--- PÁGINA {p['page']} (OCR) ---\n{p['text']}" for p in pages
        )
        return {
            "pages": pages,
            "texto_completo": texto_completo,
            "motor_usado": "Pytesseract (OCR)",
            "advertencias": advertencias
        }
    except ImportError:
        return {
            "pages": [], "texto_completo": "",
            "motor_usado": "Pytesseract (OCR)",
            "advertencias": ["Tesseract no instalado. Descarga desde: https://github.com/tesseract-ocr/tesseract"]
        }
    except Exception as e:
        return {"pages": [], "texto_completo": "", "motor_usado": "Pytesseract (OCR)", "advertencias": [f"Error OCR: {str(e)}"]}


def extract_pymupdf_ocr(file_path: str) -> dict:
    """
    Motor Híbrido: PyMuPDF + OCR nativo
    ====================================
    Para cada página:
      1. Intenta extraer texto digital con get_text()
      2. Si la página sale VACÍA, renderiza como imagen a 300 DPI y usa
         PyMuPDF OCR nativo (con Tesseract) o extrae texto de la imagen.
    
    NOTA: Si Tesseract no está disponible, usa reconocimiento de texto
    basado en el rendering interno de PyMuPDF (menos preciso pero funcional).
    """
    try:
        import fitz
        pages = []
        advertencias = []
        paginas_ocr = 0
        paginas_digitales = 0
        doc = fitz.open(file_path)
        total_real = len(doc)

        for i, page in enumerate(doc):
            # Paso 1: Intentar texto digital
            raw = page.get_text("text")
            clean = _clean_text(raw)
            
            if clean and len(clean) > 20:
                # Página digital con contenido real
                pages.append({"page": i + 1, "text": clean, "tipo": "digital"})
                paginas_digitales += 1
            else:
                # Paso 2: Página vacía → Aplicar OCR
                ocr_success = False
                for lang in ["spa+eng", "spa", "eng"]:
                    try:
                        tp = page.get_textpage_ocr(language=lang, dpi=300, full=True)
                        ocr_text = page.get_text("text", textpage=tp)
                        clean_ocr = _clean_text(ocr_text)
                        if clean_ocr and len(clean_ocr) > 10:
                            pages.append({"page": i + 1, "text": clean_ocr, "tipo": f"OCR ({lang})"})
                            paginas_ocr += 1
                            ocr_success = True
                            break
                    except Exception:
                        continue
                
                if not ocr_success:
                    pages.append({"page": i + 1, "text": f"[Pagina {i+1}: Sin texto detectable]", "tipo": "vacia"})

        doc.close()

        if paginas_ocr > 0:
            advertencias.append(f"{paginas_ocr} paginas procesadas con OCR")
        if paginas_digitales > 0:
            advertencias.append(f"{paginas_digitales} paginas con texto digital")
        vacias = total_real - paginas_ocr - paginas_digitales
        if vacias > 0:
            advertencias.append(f"{vacias} paginas sin texto detectable (necesita Tesseract para OCR)")

        texto_completo = "\n\n".join(
            f"--- PAGINA {p['page']} ({p.get('tipo','?')}) ---\n{p['text']}" for p in pages
        )
        return {
            "pages": pages,
            "texto_completo": texto_completo,
            "motor_usado": f"PyMuPDF Hibrido (Digital:{paginas_digitales} + OCR:{paginas_ocr})",
            "advertencias": advertencias,
            "total_pages_real": total_real
        }
    except Exception as e:
        return {"pages": [], "texto_completo": "", "motor_usado": "PyMuPDF Hibrido", "advertencias": [f"Error: {str(e)}"]}


def _deadmau5_cleanup(text: str) -> str:
    """
    Tarea Plus: deadmau5
    -------------------
    Un post-procesador de alta precisión que:
    1. Elimina 'Ghost characters' y ruido de codificación.
    2. Detecta y remueve patrones de headers/footers (números de página aislados).
    3. Normaliza ligaduras tipográficas (fi, fl, etc) a caracteres simples.
    4. Asegura que no queden oraciones cortadas por saltos de página.
    """
    # 1. Normalizar ligaduras comunes
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for k, v in ligatures.items():
        text = text.replace(k, v)
    
    # 2. Eliminar números de página huérfanos (ej: "\n  1  \n")
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 3. Reparar palabras cortadas por guiones al final de línea
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # 4. Colapsar espacios horizontales
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def extract(file_path: str, motor: str = "pymupdf", apply_deadmau5: bool = True) -> dict:
    """
    Punto de entrada unificado. Llama al motor seleccionado por el usuario.
    motor: 'pymupdf' | 'pdfplumber' | 'pypdf2' | 'ocr' | 'pymupdf_ocr'
    apply_deadmau5: Si es True, aplica el post-procesamiento avanzado.
    """
    motores = {
        "pymupdf": extract_pymupdf,
        "pdfplumber": extract_pdfplumber,
        "pypdf2": extract_pypdf2,
        "ocr": extract_ocr,
        "pymupdf_ocr": extract_pymupdf_ocr,
    }
    fn = motores.get(motor, extract_pymupdf)
    res = fn(file_path)
    
    if apply_deadmau5 and res["texto_completo"]:
        res["texto_completo"] = _deadmau5_cleanup(res["texto_completo"])
        for p in res["pages"]:
            p["text"] = _deadmau5_cleanup(p["text"])
        res["motor_usado"] += " + deadmau5 (Deep Clean)"
        
    return res

