// static/script.js (Updated for New Form Fields)

document.addEventListener('DOMContentLoaded', () => {
    // --- Get Elements ---
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
    const processingTimeSpan = document.getElementById('processing-time');
    const errorArea = document.getElementById('error-area');
    const downloadButton = document.getElementById('download-button');

    // --- Event Listeners ---

    // Update grade display
    if (gradeInput && gradeValueSpan) {
        gradeInput.addEventListener('input', () => {
            gradeValueSpan.textContent = gradeInput.value;
        });
        // Initial display
        gradeValueSpan.textContent = gradeInput.value;
    } else {
        console.error("Grade input or display span not found.");
    }

    // Form submission handler
    if (form) {
        form.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default page reload

            const pdfFile = pdfFileInput?.files[0];
            if (!pdfFile) {
                showError("Please select a PDF file.");
                return;
            }
            if (!pdfFile.name.toLowerCase().endsWith('.pdf')) {
                 showError("Invalid file type. Please select a PDF file.");
                return;
            }

            // --- UI Updates: Start Loading ---
            showLoading(true);
            showError(null); // Clear previous errors
            summaryOutputDiv.style.display = 'none'; // Hide previous summary
            downloadButton.style.display = 'none'; // Hide download button
            updateStatus("Uploading and processing PDF... This may take several minutes depending on the PDF size and options selected.");

            // --- Prepare Form Data ---
            const formData = new FormData();
            formData.append('pdfFile', pdfFile);
            formData.append('grade', gradeInput.value);

            // Get selected duration
            const durationElement = document.querySelector('input[name="duration"]:checked');
            if (durationElement) {
                 formData.append('duration', durationElement.value);
            } else {
                 formData.append('duration', '20'); // Default if none selected (shouldn't happen with 'checked')
                 console.warn("No duration selected, defaulting to 20.");
            }

            // Get toggle values (send 'true' or 'false' strings)
            formData.append('ocr', document.getElementById('ocr')?.checked ?? 'false');
            formData.append('mathHandling', document.getElementById('mathHandling')?.checked ?? 'true'); // Default true
            formData.append('sentenceCompletion', document.getElementById('sentenceCompletion')?.checked ?? 'true'); // Default true
            formData.append('deduplication', document.getElementById('deduplication')?.checked ?? 'true'); // Default true

            // Get advanced options
            formData.append('chunkSize', document.getElementById('chunkSize')?.value ?? '800'); // Default 800
            formData.append('overlap', document.getElementById('overlap')?.value ?? '75'); // Default 75

            // --- Make API Call ---
            try {
                updateStatus("Processing with AI model... Please wait."); // Update status during processing
                const response = await fetch('/summarize', {
                    method: 'POST',
                    body: formData
                    // Add timeout signal if needed for very long requests
                    // signal: AbortSignal.timeout(600000) // Example: 10 minute timeout
                });

                // Check if response is JSON, otherwise show raw text
                const contentType = response.headers.get("content-type");
                let result;
                 if (contentType && contentType.indexOf("application/json") !== -1) {
                     result = await response.json();
                 } else {
                     const textResponse = await response.text();
                     throw new Error(`Received non-JSON response from server: ${response.status} ${response.statusText}. Response: ${textResponse}`);
                 }


                if (!response.ok || result.error) {
                    throw new Error(result.error || `Server error: ${response.statusText}`);
                }

                // --- UI Updates: Success ---
                updateStatus("Summary generated successfully!");
                summaryTextArea.value = result.summary || "No summary content received."; // Handle empty summary
                wordCountSpan.textContent = result.word_count || '0';
                processingTimeSpan.textContent = result.processing_time?.toFixed(2) || 'N/A'; // Format time
                summaryOutputDiv.style.display = 'block'; // Show output area
                setupDownload(result.summary || "", pdfFile.name); // Setup download link
                downloadButton.style.display = 'block'; // Show download button

            } catch (error) {
                console.error("Summarization Error:", error);
                // Check for specific error types if needed (e.g., AbortError for timeout)
                // if (error.name === 'AbortError') {
                //     showError("Request timed out. The PDF might be too large or complex for the selected settings.");
                // } else {
                    showError(`Error: ${error.message}`);
                // }
                updateStatus("Failed to generate summary.");

            } finally {
                // --- UI Updates: End Loading ---
                showLoading(false);
            }
        });
    } else {
        console.error("Summary form not found.");
    }


    // --- Helper Functions ---
    function updateStatus(message) {
        if (statusMessage) {
            statusMessage.textContent = message;
        }
    }

    function showLoading(isLoading) {
        if (loader) {
            loader.style.display = isLoading ? 'block' : 'none';
        }
         if (submitButton) {
            submitButton.disabled = isLoading;
             submitButton.textContent = isLoading ? 'Processing...' : '✨ Generate Summary ✨';
        }
    }

    function showError(message) {
        if (errorArea) {
            errorArea.textContent = message || '';
            errorArea.style.display = message ? 'block' : 'none';
        }
    }

    let currentDownloadUrl = null; // Keep track of the Blob URL to revoke it later

    function setupDownload(summaryText, originalFilename) {
        if (!downloadButton) return;

        // Revoke the previous Blob URL if it exists to free memory
        if (currentDownloadUrl) {
            URL.revokeObjectURL(currentDownloadUrl);
            currentDownloadUrl = null;
        }

        // Create a new Blob and URL
        const blob = new Blob([summaryText], { type: 'text/plain;charset=utf-8' });
        currentDownloadUrl = URL.createObjectURL(blob);

        // Set up the click handler for the download button
        downloadButton.onclick = () => {
            const a = document.createElement('a');
            a.href = currentDownloadUrl;
            // Create a filename (e.g., original_summary_G6.txt)
            const baseName = originalFilename.replace(/\.pdf$/i, '');
            const gradeValue = gradeInput?.value ?? 'N'; // Use N if grade input not found
            a.download = `${baseName}_summary_G${gradeValue}.txt`;
            // Trigger download
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            // Note: We don't revoke the URL here immediately after click,
            // allowing the user to potentially click again if needed.
            // It will be revoked when a *new* summary is generated.
        };
    }

}); // End DOMContentLoaded