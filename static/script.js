document.addEventListener('DOMContentLoaded', () => {
    const videoUpload = document.getElementById('video-upload');
    const submitBtn = document.getElementById('submit-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.querySelector('.progress');
    const statusText = document.getElementById('status-text');
    const resultContainer = document.getElementById('result-container');
    const downloadLink = document.getElementById('download-link');

    let selectedFile = null;

    videoUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            document.querySelector('.upload-btn').textContent = selectedFile.name;
        }
    });

    submitBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please select a video file first.');
            return;
        }

        progressContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        progressBar.style.width = '0%';
        statusText.textContent = 'Uploading and processing...';

        const formData = new FormData();
        formData.append('video', selectedFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                progressBar.style.width = '100%';
                statusText.textContent = 'Processing complete!';
                downloadLink.href = result.downloadUrl;
                resultContainer.classList.remove('hidden');
            } else {
                throw new Error(result.error || 'An unknown error occurred.');
            }
        } catch (error) {
            statusText.textContent = `Error: ${error.message}`;
            progressBar.style.backgroundColor = '#fa383e';
        }
    });
});
