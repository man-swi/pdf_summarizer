document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('summary-form');
    const pdfFileInput = document.getElementById('pdfFile');
    const gradeInput = document.getElementById('grade');
    const gradeValueSpan = document.getElementById('grade-value');
    const submitButton = document.getElementById('submit-button');
    const statusArea = document.getElementById('status-area');
    const statusMessage = document.getElementById('status-message');
    const loader = document.getElementById('loader');
    const summaryOutputDiv = document.getElementById('summary-output');
    const summaryTextArea = document.getElementById('summary-text');
    const wordCountSpan = document.getElementById('word-count');
    const processingTimeSpan = document.getElementById('processing-time'); // Get time span
    const errorArea = document.getElementById('error-area');
    const downloadButton = document.getElementById('download-button');

    gradeInput.addEventListener('input', () => { gradeValueSpan.textContent = gradeInput.value; });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const pdfFile = pdfFileInput.files[0];
        if (!pdfFile) { showError("Please select a PDF file."); return; }

        showLoading(true); showError(null); summaryOutputDiv.style.display = 'none';
        downloadButton.style.display = 'none'; updateStatus("Uploading and processing PDF...");

        const formData = new FormData();
        formData.append('pdfFile', pdfFile);
        formData.append('grade', gradeInput.value);
        formData.append('duration', document.querySelector('input[name="duration"]:checked').value);
        formData.append('ocr', document.getElementById('ocr').checked);
        formData.append('chunkSize', document.getElementById('chunkSize').value);
        formData.append('overlap', document.getElementById('overlap').value);
        // <<< NEW: Append refinement options >>>
        formData.append('sentenceCompletion', document.getElementById('sentenceCompletion').checked);
        formData.append('deduplication', document.getElementById('deduplication').checked);

        try {
            const response = await fetch('/summarize', { method: 'POST', body: formData });
            const result = await response.json();

            if (!response.ok || result.error) { throw new Error(result.error || `Server error: ${response.statusText}`); }

            updateStatus("Summary generated successfully!");
            summaryTextArea.value = result.summary;
            wordCountSpan.textContent = result.word_count;
            processingTimeSpan.textContent = result.processing_time || 'N/A'; // Display time
            summaryOutputDiv.style.display = 'block';
            setupDownload(result.summary, pdfFile.name);
            downloadButton.style.display = 'block';

        } catch (error) {
            console.error("Summarization Error:", error);
            showError(`Error: ${error.message}`);
            updateStatus("Failed to generate summary.");
        } finally { showLoading(false); }
    });

    function updateStatus(message) { statusMessage.textContent = message; }
    function showLoading(isLoading) { loader.style.display = isLoading ? 'block' : 'none'; submitButton.disabled = isLoading; }
    function showError(message) { errorArea.textContent = message || ''; errorArea.style.display = message ? 'block' : 'none'; }

    let currentDownloadUrl = null; // Keep track of the URL
    function setupDownload(summaryText, originalFilename) {
        if (currentDownloadUrl) { URL.revokeObjectURL(currentDownloadUrl); } // Revoke previous URL
        const blob = new Blob([summaryText], { type: 'text/plain;charset=utf-8' });
        currentDownloadUrl = URL.createObjectURL(blob); // Assign new URL
        downloadButton.onclick = () => {
            const a = document.createElement('a');
            a.href = currentDownloadUrl;
            const baseName = originalFilename.replace(/\.pdf$/i, '');
            a.download = `${baseName}_summary_G${gradeInput.value}.txt`;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
        };
    }
}); // End DOMContentLoaded