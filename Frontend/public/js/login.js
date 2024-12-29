document.getElementById('loginForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const data = {
        username: username,
        password: password
    };

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        if (response.ok) {
            window.location.href = '/user';
        } else {
            document.getElementById('message').innerText = result.error || 'Login failed';
            document.getElementById('message').style = "visibility: visible;"
        }
    } catch (error) {
        document.getElementById('errorMessage').innerText = 'Error occurred during login. Please try again.';
        console.error('Error during login:', error);
    }
});

