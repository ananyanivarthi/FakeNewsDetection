const form = document.getElementById('news-form');
const resultDiv = document.getElementById('result');
const predictionP = document.getElementById('prediction');

form.addEventListener('submit', (event) => {
    event.preventDefault();
    const userInput = document.getElementById('userInput').value;
    const formData = new FormData(form);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.text())
        .then(data => {
            resultDiv.style.display = 'block';
            predictionP.textContent = data;
        })
        .catch(error => {
            console.error('Error:', error);
        });
});
