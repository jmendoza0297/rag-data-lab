"""
normative_extractor.py — Extractor Especializado para Documentos Normativos CACES
==================================================================================
Analiza PDFs de normativas educativas (CACES, CES, etc.) y extrae una estructura
jerárquica de nodos que preserva la relación padre-hijo entre Criterios e Indicadores.

Genera un JSON estructurado con:
  - Metadata del documento (institución, versión, fecha)
  - Nodos jerárquicos con relaciones padre-hijo
  - Clasificación de indicadores (Cualitativo/Cuantitativo)
  - Fórmulas de cálculo (textual)
  - Estándares de calidad
  - Referencias legales (LOES, RLOES, RFTT, RRA)
  - Elementos Fundamentales con orden de relevancia
  - Evidencias requeridas

Diseñado para integración con el pipeline RAG de LlamaIndex + ChromaDB.
"""

import re
import json
from typing import Optional


# ==============================================================================
# PATRONES REGEX PARA DOCUMENTOS NORMATIVOS ECUATORIANOS
# ==============================================================================

# Detecta headings Markdown: ## **5.2.1 Indicador Nombre...**
PAT_HEADING = re.compile(
    r"^#{1,4}\s+\*{0,2}"                     # ## o ### con opcionales **
    r"(\d{1,2}(?:\.\d{1,2}){0,3})"           # Número jerárquico: 5, 5.2, 5.2.1
    r"\.?\s+"                                  # Punto/espacio opcional
    r"(.*?)\*{0,2}\s*$",                       # Título (sin asteriscos al final)
    re.MULTILINE
)

# Detecta secciones de nivel 1 sin numeración: ## **Estándar**, ## **Descripción**
PAT_SUBSECTION = re.compile(
    r"^#{1,4}\s+\*{0,2}"
    r"(Estándar|Est[áa]ndar de [Cc]alidad|Descripci[oó]n|"
    r"Forma de C[áa]lculo|F[óo]rmula de C[áa]lculo|"
    r"Donde\s*:|"
    r"Elementos [Ff]undamentales.*|"
    r"Evidencias|"
    r"Tipo de Indicador.*|"
    r"Per[ií]odo de Evaluaci[oó]n.*)"
    r"\*{0,2}\s*:?\s*(.*)?$",
    re.MULTILINE | re.IGNORECASE
)

# Referencias legales: LOES Art. 93, RLOES Art 43, RFTT Artículo 21, RRA Art. 11
PAT_LEGAL_REF = re.compile(
    r"((?:LOES|RLOES|RFTT|RRA|CES|CACES)\s+Art[íi]?c?u?l?o?\.?\s*\d+(?:\.\d+)?)",
    re.IGNORECASE
)

# Fórmulas: FPEI = 100 * (PFPEI / TP)
PAT_FORMULA = re.compile(
    r"([A-Z]{2,}\s*=\s*(?:\d+\s*[×x\*]\s*)?(?:\(.*?\)|[A-Z/\s\d\+\-\*]+))",
    re.IGNORECASE
)

# Tipo de indicador — acepta **Tipo de Indicador** : Cualitativo  y  Tipo de Indicador: Cuantitativo
PAT_TIPO_INDICADOR = re.compile(
    r"\*{0,2}Tipo\s+de\s+Indicador\*{0,2}\s*:?\s*:?\s*\*{0,2}\s*(Cualitativo|Cuantitativo)",
    re.IGNORECASE
)

# Tabla de utilidad / valoración (valores 0, 0.35, 0.7, 1)
PAT_TABLA_UTILIDAD = re.compile(
    r"(0[,.]00|0[,.]35|0[,.]70?|1[,.]00?)",
)


def _determine_nivel(id_jerarquico: str) -> int:
    """Determina el nivel de profundidad: 1=Estructura, 2=Criterio, 3=Indicador, 4=Sub-indicador."""
    parts = id_jerarquico.split(".")
    return len(parts)


def _determine_padre(id_jerarquico: str) -> Optional[str]:
    """Retorna el ID del padre. Ej: '5.2.1' -> '5.2', '5.2' -> '5', '5' -> None."""
    parts = id_jerarquico.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def _extract_legal_refs(text: str) -> list:
    """Extrae todas las referencias legales únicas del texto."""
    refs = PAT_LEGAL_REF.findall(text)
    # Normalizar y deduplicar
    seen = set()
    result = []
    for ref in refs:
        normalized = re.sub(r"\s+", " ", ref.strip())
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def _extract_formula(text: str) -> Optional[str]:
    """Extrae fórmulas de cálculo del texto."""
    matches = PAT_FORMULA.findall(text)
    if matches:
        return matches[0].strip()
    return None


def _clean_title(title: str) -> str:
    """Limpia el título removiendo marcadores Markdown y espacios extra."""
    title = re.sub(r"\*{1,2}", "", title)  # Remover negritas markdown
    title = re.sub(r"\s+", " ", title)     # Colapsar espacios
    return title.strip()


def extract_normative_structure(full_text: str, document_metadata: dict = None) -> dict:
    """
    Extrae la estructura jerárquica completa de un documento normativo.

    Args:
        full_text: Texto completo del PDF (ya extraído por el motor de extracción)
        document_metadata: Metadata base del documento (institución, nombre, etc.)

    Returns:
        dict con estructura:
        {
            "metadata": {...},
            "nodes": [...],
            "stats": {...}
        }
    """
    if document_metadata is None:
        document_metadata = {}

    lines = full_text.split("\n")
    nodes = []
    current_node = None
    current_subsection = None  # Estándar, Descripción, Evidencias, etc.
    content_buffer = []

    # Estado para tracking
    stats = {
        "total_nodos": 0,
        "criterios": 0,
        "indicadores": 0,
        "cualitativos": 0,
        "cuantitativos": 0,
        "formulas_detectadas": 0,
        "refs_legales_unicas": set(),
        "tablas_utilidad": 0,
    }

    def _flush_content():
        """Guarda el contenido acumulado en el nodo actual."""
        nonlocal content_buffer, current_subsection
        if current_node and content_buffer:
            text = "\n".join(content_buffer).strip()
            if text:
                if current_subsection:
                    key = current_subsection.lower().replace(" ", "_")
                    # Mapear nombres de subsecciones a keys consistentes
                    key_map = {
                        "estándar": "estandar",
                        "estandar": "estandar",
                        "estándar_de_calidad": "estandar",
                        "estandar_de_calidad": "estandar",
                        "descripción": "descripcion",
                        "descripcion": "descripcion",
                        "forma_de_cálculo": "formula_calculo",
                        "forma_de_calculo": "formula_calculo",
                        "fórmula_de_cálculo": "formula_calculo",
                        "formula_de_calculo": "formula_calculo",
                        "donde:": "formula_variables",
                        "donde": "formula_variables",
                        "evidencias": "evidencias",
                    }
                    key = key_map.get(key, key)

                    if key.startswith("elementos_fundamentales"):
                        key = "elementos_fundamentales"

                    if key.startswith("tipo_de_indicador"):
                        # Extraer tipo
                        tipo_match = PAT_TIPO_INDICADOR.search(text)
                        if tipo_match:
                            current_node["tipo_indicador"] = tipo_match.group(1).capitalize()
                            if "cuantitativo" in tipo_match.group(1).lower():
                                stats["cuantitativos"] += 1
                            else:
                                stats["cualitativos"] += 1
                        return  # No guardar como contenido

                    if key.startswith("período_de_evaluación") or key.startswith("periodo_de_evaluación"):
                        current_node["periodo_evaluacion"] = text
                        return

                    # Guardar contenido de la subsección
                    if key == "formula_calculo":
                        # Intentar extraer la fórmula limpia
                        formula = _extract_formula(text)
                        current_node["formula"] = formula or text
                        stats["formulas_detectadas"] += 1
                    elif key == "formula_variables":
                        current_node["formula_variables"] = text
                    elif key == "estandar":
                        current_node["estandar"] = text
                    elif key == "descripcion":
                        current_node["descripcion"] = text
                    elif key == "evidencias":
                        # Parsear lista de evidencias
                        evidencias = []
                        for line in text.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("##"):
                                # Remover numeración y bullets
                                clean = re.sub(r"^[\d]+[.)]\s*|^[-•]\s*|^[a-z]\)\s*", "", line).strip()
                                if clean and len(clean) > 5:
                                    evidencias.append(clean)
                        current_node["evidencias"] = evidencias
                    elif key == "elementos_fundamentales":
                        # Parsear elementos con orden
                        elementos = []
                        for line in text.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("##"):
                                match = re.match(r"^(\d+)[.)]\s*(.*)", line)
                                if match:
                                    elementos.append({
                                        "orden": int(match.group(1)),
                                        "texto": match.group(2).strip()
                                    })
                                else:
                                    clean = re.sub(r"^[-•]\s*", "", line).strip()
                                    if clean and len(clean) > 5:
                                        if elementos:
                                            # Anexar al último elemento
                                            elementos[-1]["texto"] += " " + clean
                                        else:
                                            elementos.append({"orden": len(elementos)+1, "texto": clean})
                        current_node["elementos_fundamentales"] = elementos
                    else:
                        # Guardar como contenido genérico de la subsección
                        if "contenido_secciones" not in current_node:
                            current_node["contenido_secciones"] = {}
                        current_node["contenido_secciones"][key] = text
                else:
                    # Contenido directo del nodo (no subsección)
                    if "contenido" not in current_node or not current_node["contenido"]:
                        current_node["contenido"] = text
                    else:
                        current_node["contenido"] += "\n" + text

        content_buffer = []

    def _finalize_node():
        """Finaliza el nodo actual y lo agrega a la lista."""
        nonlocal current_node, current_subsection
        _flush_content()
        if current_node:
            # Extraer referencias legales del contenido completo
            all_text = json.dumps(current_node, ensure_ascii=False)
            legal_refs = _extract_legal_refs(all_text)
            if legal_refs:
                current_node["sustento_legal"] = legal_refs
                stats["refs_legales_unicas"].update(legal_refs)

            # Detectar si tiene tabla de utilidad
            if PAT_TABLA_UTILIDAD.search(all_text):
                current_node["tiene_tabla_utilidad"] = True
                stats["tablas_utilidad"] += 1

            # Clasificar nivel semántico
            nivel = current_node.get("nivel", 1)
            if nivel == 2:
                current_node["tipo_nodo"] = "criterio"
                stats["criterios"] += 1
            elif nivel >= 3 and "indicador" in current_node.get("titulo", "").lower():
                current_node["tipo_nodo"] = "indicador"
                stats["indicadores"] += 1
            else:
                current_node["tipo_nodo"] = "seccion"

            nodes.append(current_node)
            stats["total_nodos"] += 1
        current_node = None
        current_subsection = None

    # === PARSER PRINCIPAL: Recorrer líneas ===
    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Skip líneas vacías
        if not line_stripped:
            if content_buffer:
                content_buffer.append("")  # Preservar párrafos
            continue

        # Skip headers repetitivos del PDF (encabezado de página)
        if "MODELO Y METODOLOGÍA" in line_stripped or "intentionally omitted" in line_stripped:
            continue

        # ¿Es un heading jerárquico? (## **5.2.1 Indicador ...** )
        heading_match = PAT_HEADING.match(line_stripped)
        if heading_match:
            id_jer = heading_match.group(1)
            titulo_raw = heading_match.group(2)
            titulo = _clean_title(titulo_raw)

            # Solo procesar si parece un nodo real del modelo CACES.
            # Filtro estricto: evitar sub-headings internos como "1. Datos Generales"
            # que están DENTRO de la descripción de un indicador.
            parts = id_jer.split(".")
            is_real_node = False

            if "criterio" in titulo.lower() or "indicador" in titulo.lower():
                # Siempre un nodo real si menciona "Criterio" o "Indicador"
                is_real_node = True
            elif len(parts) == 1:
                first_num = int(parts[0])
                # Solo nodos de nivel 1 si la numeración es <= 10 y NO estamos
                # dentro de un indicador (donde aparecen sub-headings 1-8)
                if current_node and "indicador" in current_node.get("titulo", "").lower():
                    # Estamos dentro de un indicador — NO crear nodo nuevo
                    is_real_node = False
                elif first_num <= 10:
                    is_real_node = True
            elif len(parts) == 2:
                # X.X → es un criterio (5.1, 5.2, etc.) si tiene texto alfanumérico
                is_real_node = any(c.isalpha() for c in titulo)
            elif len(parts) == 3:
                # X.X.X → indicador
                is_real_node = True

            if is_real_node:
                _finalize_node()
                current_subsection = None
                current_node = {
                    "id_jerarquico": id_jer,
                    "padre": _determine_padre(id_jer),
                    "nivel": _determine_nivel(id_jer),
                    "titulo": titulo,
                    "contenido": "",
                }
                continue

        # ¿Es una subsección interna? (## **Estándar**, ## **Descripción**, etc.)
        subsection_match = PAT_SUBSECTION.match(line_stripped)
        if subsection_match and current_node:
            _flush_content()
            subsection_name = subsection_match.group(1).strip()
            inline_value = subsection_match.group(2) if subsection_match.group(2) else ""

            current_subsection = subsection_name

            # Si tiene valor inline (ej: "Tipo de Indicador: Cuantitativo")
            if inline_value.strip():
                content_buffer.append(inline_value.strip())

            # Caso especial: Tipo de indicador en la misma línea
            tipo_match = PAT_TIPO_INDICADOR.search(line_stripped)
            if tipo_match:
                current_node["tipo_indicador"] = tipo_match.group(1).capitalize()
                if "cuantitativo" in tipo_match.group(1).lower():
                    stats["cuantitativos"] += 1
                else:
                    stats["cualitativos"] += 1
                current_subsection = None  # Reset, ya procesado
            continue

        # ¿Tiene "Tipo de Indicador" como texto normal (sin heading ##)?
        if current_node and not current_subsection:
            tipo_match = PAT_TIPO_INDICADOR.search(line_stripped)
            if tipo_match and "tipo_indicador" not in current_node:
                current_node["tipo_indicador"] = tipo_match.group(1).capitalize()
                if "cuantitativo" in tipo_match.group(1).lower():
                    stats["cuantitativos"] += 1
                else:
                    stats["cualitativos"] += 1
                continue

        # Línea de contenido normal
        if current_node:
            # Limpiar marcadores markdown
            clean_line = re.sub(r"^\*{0,2}(.*?)\*{0,2}$", r"\1", line_stripped)
            clean_line = re.sub(r"^[-•]\s*", "- ", clean_line)  # Normalizar bullets
            content_buffer.append(clean_line)

    # Finalizar último nodo
    _finalize_node()

    # Convertir set a list para serialización
    stats["refs_legales_unicas"] = sorted(list(stats["refs_legales_unicas"]))

    return {
        "metadata": {
            "institucion": document_metadata.get("institucion", "CACES"),
            "documento": document_metadata.get("documento",
                "Modelo de Cualificación de Posgrados Tecnológicos"),
            "fecha_version": document_metadata.get("fecha_version", "Diciembre 2025"),
            "extractor": "normative_extractor v1.0",
            "total_nodos": stats["total_nodos"],
        },
        "nodes": nodes,
        "stats": stats,
    }


def extract_and_serialize(full_text: str, output_path: str = None, document_metadata: dict = None) -> dict:
    """
    Extrae la estructura y opcionalmente la guarda como JSON.

    Args:
        full_text: Texto completo del PDF
        output_path: Ruta para guardar el JSON (opcional)
        document_metadata: Metadata del documento

    Returns:
        dict con la estructura completa
    """
    result = extract_normative_structure(full_text, document_metadata)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ==============================================================================
# TEST STANDALONE
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Extraer texto del PDF
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    pdf_path = r"media\pdfs\MODELO-Y-METODOLOGIA-DE-CUALIFICACION-ACADEMICA-08DIC2025.pdf"

    print(f"Extrayendo texto de: {pdf_path}")
    import pymupdf4llm
    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    full_text = "\n".join(c.get("text", "") for c in page_chunks)

    print(f"Texto extraído: {len(full_text)} caracteres")
    print("Ejecutando extractor normativo...")

    result = extract_and_serialize(
        full_text,
        output_path="normative_output.json",
        document_metadata={
            "institucion": "CACES",
            "documento": "Modelo y Metodología de Cualificación Académica - Posgrados Tecnológicos",
            "fecha_version": "Diciembre 2025",
        }
    )

    print(f"\n{'='*60}")
    print(f"RESULTADOS:")
    print(f"{'='*60}")
    print(f"  Total nodos: {result['stats']['total_nodos']}")
    print(f"  Criterios: {result['stats']['criterios']}")
    print(f"  Indicadores: {result['stats']['indicadores']}")
    print(f"  Cualitativos: {result['stats']['cualitativos']}")
    print(f"  Cuantitativos: {result['stats']['cuantitativos']}")
    print(f"  Fórmulas: {result['stats']['formulas_detectadas']}")
    print(f"  Tablas utilidad: {result['stats']['tablas_utilidad']}")
    print(f"  Refs legales únicas: {len(result['stats']['refs_legales_unicas'])}")

    print(f"\nReferencias Legales:")
    for ref in result['stats']['refs_legales_unicas']:
        print(f"    - {ref}")

    print(f"\nPrimeros 5 nodos:")
    for node in result['nodes'][:5]:
        print(f"\n  [{node['id_jerarquico']}] {node['titulo']}")
        print(f"     Nivel: {node['nivel']} | Padre: {node.get('padre', 'ROOT')}")
        print(f"     Tipo: {node.get('tipo_nodo', '?')} | Indicador: {node.get('tipo_indicador', 'N/A')}")
        if node.get('estandar'):
            print(f"     Estándar: {node['estandar'][:120]}...")
        if node.get('formula'):
            print(f"     Fórmula: {node['formula']}")
        if node.get('sustento_legal'):
            print(f"     Legal: {', '.join(node['sustento_legal'])}")

    print(f"\n  JSON completo guardado en: normative_output.json")
