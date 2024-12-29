document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("register-form");
    const message = document.getElementById("message");

    form.addEventListener("submit", async (event) => {
        event.preventDefault(); // Prevent page reload

        // Gather form data
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value.trim();

        // Validate inputs
        if (!username || !password) {
            showMessage("Please fill in all fields.", "error");
            return;
        }

        try {
            // Send registration data to API
            const response = await fetch("/api/register", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username,
                    password,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                showMessage(data.message || "Registration successful!", "success");

                // Update the message to 'Redirecting...' after 2 seconds
                setTimeout(() => {
                    showMessage("Redirecting...", "success");
                }, 2000);

                // Redirect to login page after 5 seconds
                setTimeout(() => {
                    window.location.href = "login.html";
                }, 5000);
            } else {
                const errorData = await response.json();
                showMessage(errorData.message || "Registration failed.", "error");
            }
        } catch (error) {
            console.error("Error during registration:", error);
            showMessage("An unexpected error occurred. Please try again later.", "error");
        }
    });

    // Function to display messages
    function showMessage(text, type) {
        message.innerText = text;
        message.style.visibility = "visible"; // Ensure the message is visible
        message.style.color = type === "success" ? "lightgreen" : "red";
    }
});
