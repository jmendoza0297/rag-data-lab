filepath = r"c:\Users\PC-EURO\.gemini\antigravity-ide\scratch\rag_django_ollama\api\templates\index.html"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. replace variables
old_vars = "        let currentDocId = null;"
new_vars = """        let currentDocId = null;
        let allDocuments = [];"""

# 2. replace HTML config in L258-L275
old_html_config = """                    <!-- Configuración dinámica según modo -->
                    <div id="directConfig" style="display:flex; align-items:center; gap:4px;">
                        <span>Top K:</span>
                        <select id="vectorK" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--text-main); font-size:0.75rem;">
                            <option value="3">3</option><option value="5" selected>5</option><option value="10">10</option><option value="20">20</option>
                        </select>
                    </div>
                    
                    <div id="routerConfig" style="display:none; align-items:center; gap:8px;">
                        <span>Modelo IA:</span>
                        <select id="routerModel" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--accent); font-weight:600; font-size:0.75rem;">
                            <option value="llama3">Llama 3 (8B)</option>
                            <option value="qwen2:1.5b" selected>Qwen 2 (1.5B)</option>
                            <option value="llama3.1">Llama 3.1</option>
                            <option value="gemma2">Gemma 2</option>
                        </select>
                    </div>"""

new_html_config = """                    <!-- Configuración dinámica según modo -->
                    <div id="directConfig" style="display:flex; align-items:center; gap:12px;">
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span>Colección Qdrant:</span>
                            <select id="directCollection" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--text-main); font-size:0.75rem; max-width:250px;">
                                <option value="">-- Colección del Doc Activo --</option>
                            </select>
                        </div>
                        <div style="display:flex; align-items:center; gap:4px;">
                            <span>Top K:</span>
                            <select id="vectorK" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--text-main); font-size:0.75rem;">
                                <option value="3">3</option><option value="5" selected>5</option><option value="10">10</option><option value="20">20</option>
                            </select>
                        </div>
                    </div>
                    
                    <div id="routerConfig" style="display:none; align-items:center; gap:12px;">
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span>Colección Qdrant:</span>
                            <select id="routerCollection" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--text-main); font-size:0.75rem; max-width:250px;">
                                <option value="">-- Colección del Doc Activo --</option>
                            </select>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span>Modelo IA:</span>
                            <select id="routerModel" style="border:1px solid var(--border); border-radius:4px; padding:2px 4px; background:var(--bg-main); color:var(--accent); font-weight:600; font-size:0.75rem;">
                                <option value="llama3">Llama 3 (8B)</option>
                                <option value="qwen2:1.5b" selected>Qwen 2 (1.5B)</option>
                                <option value="llama3.1">Llama 3.1</option>
                                <option value="gemma2">Gemma 2</option>
                            </select>
                        </div>
                    </div>"""

# 3. replace fetchDocuments start
old_fetch = """        async function fetchDocuments() {
            try {
                const res = await fetch('/api/documents/');
                const docs = await res.json();
                const list = document.getElementById('docList');"""

new_fetch = """        async function fetchDocuments() {
            try {
                const res = await fetch('/api/documents/');
                const docs = await res.json();
                allDocuments = docs; // Guardar globalmente
                updateCollectionSelectors(docs); // Actualizar los dropdowns de colecciones
                const list = document.getElementById('docList');"""

# 4. replace selectDoc
old_select = """        function selectDoc(id) {
            currentDocId = id;
            fetchDocuments(); // Refresh UI selection
            loadPipelineData(id);
        }"""

new_select = """        function selectDoc(id) {
            currentDocId = id;
            fetchDocuments(); // Refresh UI selection
            loadPipelineData(id);
            
            // Actualizar selectores al cambiar de doc
            setTimeout(() => {
                const doc = allDocuments.find(d => d.id === id);
                const colName = (doc && doc.qdrant_collection_name) ? doc.qdrant_collection_name : "";
                const selectDirect = document.getElementById('directCollection');
                const selectRouter = document.getElementById('routerCollection');
                if (selectDirect) selectDirect.value = colName;
                if (selectRouter) selectRouter.value = colName;
            }, 300);
        }"""

# 5. replace vectorSearch signature / request
old_vector_search = """        async function vectorSearch() {
            if (!currentDocId) { alert("Selecciona un documento primero"); return; }
            const input = document.getElementById('vectorQueryInput');
            const query = input.value;
            if (!query) return;

            const k = document.getElementById('vectorK').value;
            const container = document.getElementById('pipeline-content');

            // Insertar resultados al inicio del pipeline
            const resultsDiv = document.getElementById('vector-results') || document.createElement('div');
            resultsDiv.id = 'vector-results';
            resultsDiv.innerHTML = '<p style="color:var(--accent);"><i>🔍 Buscando en Qdrant (172.16.21.246:6333)...</i></p>';

            if (!document.getElementById('vector-results')) {
                container.insertBefore(resultsDiv, container.firstChild);
            }

            try {
                const res = await fetch(`/api/documents/${currentDocId}/vector-search/`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, k: parseInt(k) })
                });"""

new_vector_search = """        async function vectorSearch() {
            if (!currentDocId) { alert("Selecciona un documento primero"); return; }
            const input = document.getElementById('vectorQueryInput');
            const query = input.value;
            if (!query) return;

            const k = document.getElementById('vectorK').value;
            const collectionName = document.getElementById('directCollection').value;
            const container = document.getElementById('pipeline-content');

            // Insertar resultados al inicio del pipeline
            const resultsDiv = document.getElementById('vector-results') || document.createElement('div');
            resultsDiv.id = 'vector-results';
            resultsDiv.innerHTML = '<p style="color:var(--accent);"><i>🔍 Buscando en Qdrant (172.16.21.246:6333)...</i></p>';

            if (!document.getElementById('vector-results')) {
                container.insertBefore(resultsDiv, container.firstChild);
            }

            try {
                const res = await fetch(`/api/documents/${currentDocId}/vector-search/`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, k: parseInt(k), collection_name: collectionName })
                });"""

# 6. replace intentSearch request
old_intent_search = """        async function intentSearch() {
            if (!currentDocId) { alert("Selecciona un documento primero"); return; }
            const input = document.getElementById('vectorQueryInput');
            const query = input.value;
            if (!query) return;

            const model = document.getElementById('routerModel').value;
            const container = document.getElementById('pipeline-content');

            // Insertar resultados de intent al inicio del pipeline
            const resultsDiv = document.getElementById('vector-results') || document.createElement('div');
            resultsDiv.id = 'vector-results';
            resultsDiv.innerHTML = '<p style="color:var(--accent);"><i>🧠 Clasificando intención y buscando en Qdrant con Ollama...</i></p>';

            if (!document.getElementById('vector-results')) {
                container.insertBefore(resultsDiv, container.firstChild);
            }

            try {
                const res = await fetch(`/api/documents/${currentDocId}/intent-search/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, model: model })
                });"""

new_intent_search = """        async function intentSearch() {
            if (!currentDocId) { alert("Selecciona un documento primero"); return; }
            const input = document.getElementById('vectorQueryInput');
            const query = input.value;
            if (!query) return;

            const model = document.getElementById('routerModel').value;
            const collectionName = document.getElementById('routerCollection').value;
            const container = document.getElementById('pipeline-content');

            // Insertar resultados de intent al inicio del pipeline
            const resultsDiv = document.getElementById('vector-results') || document.createElement('div');
            resultsDiv.id = 'vector-results';
            resultsDiv.innerHTML = '<p style="color:var(--accent);"><i>🧠 Clasificando intención y buscando en Qdrant con Ollama...</i></p>';

            if (!document.getElementById('vector-results')) {
                container.insertBefore(resultsDiv, container.firstChild);
            }

            try {
                const res = await fetch(`/api/documents/${currentDocId}/intent-search/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, model: model, collection_name: collectionName })
                });"""

# 7. Add updateCollectionSelectors function after setPipelineMode helper
old_helpers = """        function setPipelineMode(mode) {
            activePipelineMode = mode;
            const btnDirect = document.getElementById('modeBtnDirect');
            const btnRouter = document.getElementById('modeBtnRouter');
            const configDirect = document.getElementById('directConfig');
            const configRouter = document.getElementById('routerConfig');
            const input = document.getElementById('vectorQueryInput');
            const btnSearch = document.getElementById('pipelineSearchBtn');

            if (mode === 'direct') {
                btnDirect.style.background = 'var(--bg-panel)';
                btnDirect.style.color = 'var(--text-main)';
                btnDirect.style.boxShadow = 'var(--shadow)';
                btnRouter.style.background = 'transparent';
                btnRouter.style.color = 'var(--text-dim)';
                btnRouter.style.boxShadow = 'none';

                configDirect.style.display = 'flex';
                configRouter.style.display = 'none';
                input.placeholder = "Escribe una consulta para buscar en la base vectorial...";
                btnSearch.innerHTML = "Buscar";
            } else {
                btnDirect.style.background = 'transparent';
                btnDirect.style.color = 'var(--text-dim)';
                btnDirect.style.boxShadow = 'none';
                btnRouter.style.background = 'var(--bg-panel)';
                btnRouter.style.color = 'var(--text-main)';
                btnRouter.style.boxShadow = 'var(--shadow)';

                configDirect.style.display = 'none';
                configRouter.style.display = 'flex';
                input.placeholder = "Pregunta al Ruteador de Intenciones (Ej: 'Hazme un resumen del pdf')...";
                btnSearch.innerHTML = "Rutar & Buscar";
            }
        }"""

new_helpers = """        function setPipelineMode(mode) {
            activePipelineMode = mode;
            const btnDirect = document.getElementById('modeBtnDirect');
            const btnRouter = document.getElementById('modeBtnRouter');
            const configDirect = document.getElementById('directConfig');
            const configRouter = document.getElementById('routerConfig');
            const input = document.getElementById('vectorQueryInput');
            const btnSearch = document.getElementById('pipelineSearchBtn');

            if (mode === 'direct') {
                btnDirect.style.background = 'var(--bg-panel)';
                btnDirect.style.color = 'var(--text-main)';
                btnDirect.style.boxShadow = 'var(--shadow)';
                btnRouter.style.background = 'transparent';
                btnRouter.style.color = 'var(--text-dim)';
                btnRouter.style.boxShadow = 'none';

                configDirect.style.display = 'flex';
                configRouter.style.display = 'none';
                input.placeholder = "Escribe una consulta para buscar en la base vectorial...";
                btnSearch.innerHTML = "Buscar";
            } else {
                btnDirect.style.background = 'transparent';
                btnDirect.style.color = 'var(--text-dim)';
                btnDirect.style.boxShadow = 'none';
                btnRouter.style.background = 'var(--bg-panel)';
                btnRouter.style.color = 'var(--text-main)';
                btnRouter.style.boxShadow = 'var(--shadow)';

                configDirect.style.display = 'none';
                configRouter.style.display = 'flex';
                input.placeholder = "Pregunta al Ruteador de Intenciones (Ej: 'Hazme un resumen del pdf')...";
                btnSearch.innerHTML = "Rutar & Buscar";
            }
        }

        function updateCollectionSelectors(docs) {
            const selectDirect = document.getElementById('directCollection');
            const selectRouter = document.getElementById('routerCollection');
            if (!selectDirect || !selectRouter) return;

            const prevDirectVal = selectDirect.value;
            const prevRouterVal = selectRouter.value;

            selectDirect.innerHTML = '<option value="">-- Colección del Doc Activo --</option>';
            selectRouter.innerHTML = '<option value="">-- Colección del Doc Activo --</option>';

            docs.forEach(doc => {
                if (doc.qdrant_synced && doc.qdrant_collection_name) {
                    const cleanName = doc.nombre.length > 25 ? doc.nombre.substring(0, 25) + '...' : doc.nombre;
                    const optionText = `${cleanName} (${doc.qdrant_collection_name})`;
                    
                    const optD = document.createElement('option');
                    optD.value = doc.qdrant_collection_name;
                    optD.innerText = optionText;
                    selectDirect.appendChild(optD);

                    const optR = document.createElement('option');
                    optR.value = doc.qdrant_collection_name;
                    optR.innerText = optionText;
                    selectRouter.appendChild(optR);
                }
            });

            if (prevDirectVal && [...selectDirect.options].some(o => o.value === prevDirectVal)) {
                selectDirect.value = prevDirectVal;
            } else if (currentDocId) {
                const doc = docs.find(d => d.id === currentDocId);
                if (doc && doc.qdrant_collection_name) selectDirect.value = doc.qdrant_collection_name;
            }

            if (prevRouterVal && [...selectRouter.options].some(o => o.value === prevRouterVal)) {
                selectRouter.value = prevRouterVal;
            } else if (currentDocId) {
                const doc = docs.find(d => d.id === currentDocId);
                if (doc && doc.qdrant_collection_name) selectRouter.value = doc.qdrant_collection_name;
            }
        }"""

orig_len = len(content)

content_lf = content.replace("\r\n", "\n")
for old, new in [(old_vars, new_vars), (old_html_config, new_html_config), (old_fetch, new_fetch), (old_select, new_select), (old_vector_search, new_vector_search), (old_intent_search, new_intent_search), (old_helpers, new_helpers)]:
    old_lf = old.replace("\r\n", "\n")
    if old_lf in content_lf:
        content_lf = content_lf.replace(old_lf, new)
        print("Replaced index block successfully")
    else:
        print(f"WARNING: Could not find block in index.html starting with: {old.splitlines()[0]}")

if len(content_lf) != orig_len:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content_lf)
    print("index.html written successfully!")
else:
    print("ERROR: No replacements made to index.html")
