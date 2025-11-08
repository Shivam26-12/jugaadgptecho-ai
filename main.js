// main.js

function openGoogleMaps() {
    if (navigator.geolocation) {
        
        const locationButton = document.querySelector('#location-btn button');
        const originalText = locationButton.textContent;
        locationButton.textContent = "Locating...";
        locationButton.disabled = true;

        navigator.geolocation.getCurrentPosition(
            
            // Success Callback
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                // CORRECTED LINE (was line 25 in the previous version)
                // Correct Google Maps URL format using latitude and longitude
                const mapUrl = https://www.google.com/maps/search/?api=1&query=${lat},${lon};
                
                // Open the URL in a new browser tab
                window.open(mapUrl, '_blank');

                // Restore button state
                locationButton.textContent = originalText;
                locationButton.disabled = false;
            },
            
            // Error Callback
            (error) => {
                // ... (Error handling remains the same)
                alert("Location Error: Please ensure you allow location access.");

                // Restore button state
                locationButton.textContent = originalText;
                locationButton.disabled = false;
            }
        );
    } else {
        alert("Geolocation is not supported by your browser.");
    }
}

// NOTE: You do not need to correct line 22 in main.js. 
// Line 22 was `const mapUrl = \http://googleusercontent.com/maps.google.com/5{lat},${lon}\;\`
// The error was with the URL itself, which is corrected above.