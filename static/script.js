// static/script.js (Corrected and Cleaned)

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
        gradeValueSpan.textContent = gradeInput.value; // Initial display
    } else {
        console.error("Could not find Grade input or Grade value span element.");
    }

    // Form submission handler
    if (form) {
        form.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default page reload
            console.log("DEBUG: Form submitted!");

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
            console.log("DEBUG: Showing loading state...");
            showLoading(true);
            showError(null);
            summaryOutputDiv.style.display = 'none';
            downloadButton.style.display = 'none';
            updateStatus("Uploading and processing PDF... This may take several minutes.");

            // --- Prepare Form Data ---
            const formData = new FormData();
            console.log("DEBUG: FormData created.");
            try {
                formData.append('pdfFile', pdfFile);
                formData.append('grade', gradeInput.value);
                const durationElement = document.querySelector('input[name="duration"]:checked');
                formData.append('duration', durationElement?.value ?? '20');
                formData.append('ocr', document.getElementById('ocr')?.checked ?? false); // Send boolean as string
                formData.append('mathHandling', document.getElementById('mathHandling')?.checked ?? true);
                formData.append('sentenceCompletion', document.getElementById('sentenceCompletion')?.checked ?? true);
                formData.append('deduplication', document.getElementById('deduplication')?.checked ?? true);
                formData.append('chunkSize', document.getElementById('chunkSize')?.value ?? '800');
                formData.append('overlap', document.getElementById('overlap')?.value ?? '75');
                console.log("DEBUG: FormData populated:", Object.fromEntries(formData.entries())); // Use .entries() for logging
            } catch (e) {
                console.error("DEBUG: Error populating FormData:", e);
                showError("Error preparing form data. Check console.");
                showLoading(false);
                return;
            }

            // --- Make API Call ---
            try {
                console.log("DEBUG: Initiating fetch to /summarize...");
                updateStatus("Processing with AI model... Please wait.");
                const response = await fetch('/summarize', { // Ensure Flask route is correct
                    method: 'POST',
                    body: formData
                });
                console.log("DEBUG: Fetch response received. Status:", response.status);

                const contentType = response.headers.get("content-type");
                let result;
                 if (contentType && contentType.includes("application/json")) { // More robust check
                     result = await response.json();
                 } else {
                     const textResponse = await response.text();
                     console.error("DEBUG: Non-JSON response received:", textResponse); // Log raw response
                     throw new Error(`Server returned non-JSON response (${response.status} ${response.statusText}). Check server logs.`);
                 }

                if (!response.ok || result.error) {
                    console.error("DEBUG: Server returned error:", result?.error || `${response.status} ${response.statusText}`);
                    throw new Error(result.error || `Server error (${response.status}). Check server logs for details.`);
                }

                // --- UI Updates: Success ---
                console.log("DEBUG: Request successful. Displaying results.");
                updateStatus("Summary generated successfully!");
                summaryTextArea.value = result.summary || "[No summary content received]";
                wordCountSpan.textContent = result.word_count || '0';
                processingTimeSpan.textContent = result.processing_time?.toFixed(2) || 'N/A';
                summaryOutputDiv.style.display = 'block';
                setupDownload(result.summary || "", pdfFile.name);
                downloadButton.style.display = 'block';

            } catch (error) {
                 console.error("DEBUG: Fetch or processing failed:", error);
                 showError(`Error: ${error.message}. See browser console and server logs for details.`);
                 updateStatus("Failed to generate summary.");

            } finally {
                console.log("DEBUG: Hiding loading state.");
                showLoading(false);
            }
        });
    } else {
        console.error("Summary form element not found.");
    }

    // --- Helper Functions ---
    function updateStatus(message) {
        if (statusMessage) {
            statusMessage.textContent = message;
        } else {
            console.warn("Status message element not found.");
        }
    }

    function showLoading(isLoading) {
        if (loader) {
            loader.style.display = isLoading ? 'block' : 'none';
        } else {
             console.warn("Loader element not found.");
        }
         if (submitButton) {
            submitButton.disabled = isLoading;
             submitButton.textContent = isLoading ? 'Processing...' : '✨ Generate Summary ✨';
        } else {
             console.warn("Submit button element not found.");
        }
    }

    function showError(message) {
        if (errorArea) {
            errorArea.textContent = message || '';
            errorArea.style.display = message ? 'block' : 'none';
        } else if (message) { // Log error even if area isn't found
            console.error("Error area element not found. Error message:", message);
        }
    }

    let currentDownloadUrl = null;

    function setupDownload(summaryText, originalFilename) {
        if (!downloadButton) {
             console.warn("Download button element not found.");
             return;
        }

        if (currentDownloadUrl) {
            URL.revokeObjectURL(currentDownloadUrl);
            currentDownloadUrl = null;
        }

        const blob = new Blob([summaryText], { type: 'text/plain;charset=utf-8' });
        currentDownloadUrl = URL.createObjectURL(blob);

        downloadButton.onclick = () => {
            try {
                const a = document.createElement('a');
                a.href = currentDownloadUrl;
                const baseName = originalFilename.replace(/\.pdf$/i, '');
                const gradeValue = gradeInput?.value ?? 'N';
                a.download = `${baseName}_summary_G${gradeValue}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                // URL is revoked next time setupDownload is called
            } catch (e) {
                console.error("Error triggering download:", e);
                showError("Could not initiate download.");
            }
        };
    }

}); // End DOMContentLoaded