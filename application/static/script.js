// script.js

document.addEventListener("DOMContentLoaded", function () {
    console.log("Website loaded successfully!");

    // Simulation button functionality
    const simulationButton = document.querySelector("#startSimulation");
    const simulationResult = document.querySelector("#simulationResult");

    if (simulationButton) {
        simulationButton.addEventListener("click", function () {
            simulationResult.innerText = "Simulating optimized maritime routes...";
            setTimeout(() => {
                simulationResult.innerText = "Simulation Complete: Optimal route calculated!";
            }, 3000);
        });
    }

    // Contact Form validation
    const contactForm = document.querySelector("form");
    if (contactForm) {
        contactForm.addEventListener("submit", function (event) {
            event.preventDefault();
            alert("Thank you for reaching out! We will get back to you soon.");
        });
    }

    // Chart for route comparison
    if (document.querySelector("#routeChart")) {
        var ctx = document.getElementById("routeChart").getContext("2d");
        var chart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Optimized Route", "Minimum Distance Route"],
                datasets: [{
                    label: "Fuel Consumption (liters)",
                    data: [500, 700],
                    backgroundColor: ["blue", "red"]
                }]
            }
        });
    }
});
