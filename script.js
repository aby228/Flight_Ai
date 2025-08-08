// Flight AI - JavaScript Module
// Intelligent Flight Delay Prediction System

class FlightAI {
    constructor() {
        this.init();
    }

    // Initialize the application
    init() {
        this.setupEventListeners();
        this.setDefaultValues();
        this.updateSliderDisplays();
    }

    // Set up all event listeners
    setupEventListeners() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => this.handleFormSubmission(e));
        
        // Airport code validation
        document.getElementById('origin').addEventListener('input', this.validateAirportCode);
        document.getElementById('destination').addEventListener('input', this.validateAirportCode);
        
        // Time change listeners for auto-calculation
        document.getElementById('departureTime').addEventListener('change', () => this.updateFlightDuration());
        document.getElementById('arrivalTime').addEventListener('change', () => this.updateFlightDuration());
        
        // Slider update listeners
        document.getElementById('temperature').addEventListener('input', (e) => this.updateTempDisplay(e.target.value));
        document.getElementById('precipitation').addEventListener('input', (e) => this.updatePrecipDisplay(e.target.value));
        document.getElementById('windSpeed').addEventListener('input', (e) => this.updateWindDisplay(e.target.value));
    }

    // Set default form values
    setDefaultValues() {
        // Restrict date to January 2025 and default to a valid date (2025-01-15)
        const dateInput = document.getElementById('flightDate');
        dateInput.min = '2025-01-01';
        dateInput.max = '2025-01-31';
        dateInput.value = '2025-01-15';
        
        // Set default times (2 hours from now for departure, +3 hours for arrival)
        const now = new Date();
        const departure = new Date(now.getTime() + 2 * 60 * 60 * 1000);
        const arrival = new Date(departure.getTime() + 3 * 60 * 60 * 1000);
        
        document.getElementById('departureTime').value = departure.toTimeString().slice(0, 5);
        document.getElementById('arrivalTime').value = arrival.toTimeString().slice(0, 5);
        
        // Update flight duration
        this.updateFlightDuration();
    }

    // Update all slider displays on initialization
    updateSliderDisplays() {
        const temperature = document.getElementById('temperature').value;
        const precipitation = document.getElementById('precipitation').value;
        const windSpeed = document.getElementById('windSpeed').value;
        
        this.updateTempDisplay(temperature);
        this.updatePrecipDisplay(precipitation);
        this.updateWindDisplay(windSpeed);
    }

    // Update temperature display
    updateTempDisplay(value) {
        document.getElementById('tempDisplay').textContent = value + '°C';
    }

    // Update precipitation display
    updatePrecipDisplay(value) {
        document.getElementById('precipDisplay').textContent = value + 'mm';
    }

    // Update wind speed display
    updateWindDisplay(value) {
        document.getElementById('windDisplay').textContent = value + ' km/h';
    }

    // Validate airport codes (3 letters, uppercase only)
    validateAirportCode(e) {
        e.target.value = e.target.value.toUpperCase().replace(/[^A-Z]/g, '');
    }

    // Auto-calculate flight duration based on departure and arrival times
    updateFlightDuration() {
        const departureTime = document.getElementById('departureTime').value;
        const arrivalTime = document.getElementById('arrivalTime').value;
        
        if (departureTime && arrivalTime) {
            const [depHour, depMin] = departureTime.split(':').map(Number);
            const [arrHour, arrMin] = arrivalTime.split(':').map(Number);
            
            let duration = (arrHour * 60 + arrMin) - (depHour * 60 + depMin);
            if (duration < 0) duration += 24 * 60; // Handle overnight flights
            
            document.getElementById('flightDuration').value = (duration / 60).toFixed(1);
        }
    }

    // Handle form submission
    async handleFormSubmission(e) {
        e.preventDefault();
        
        if (!this.validateForm()) {
            return;
        }

        const btn = document.getElementById('predictBtn');
        const resultsSection = document.getElementById('resultsSection');
        
        // Show loading state
        this.showLoadingState(btn);
        
        try {
            let prediction;
            if (typeof API_URL === 'string' && API_URL.trim().length > 0) {
                prediction = await this.predictServerSide();
            } else {
                prediction = this.predictClientSide();
            }

            this.displayResults(prediction);
        } catch (err) {
            console.error('Prediction error:', err);
            this.showError('Prediction failed. Please try again.');
        } finally {
            // Reset button
            this.resetLoadingState(btn);
            // Show results with smooth scroll
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // Validate form inputs
    validateForm() {
        const origin = document.getElementById('origin').value;
        const destination = document.getElementById('destination').value;
        
        if (origin.length !== 3) {
            this.showError('Please enter a valid 3-letter origin airport code');
            return false;
        }
        
        if (destination.length !== 3) {
            this.showError('Please enter a valid 3-letter destination airport code');
            return false;
        }
        
        if (origin === destination) {
            this.showError('Origin and destination cannot be the same');
            return false;
        }
        
        return true;
    }

    // Show error message
    showError(message) {
        alert(message); // In production, use a better toast/modal system
    }

    // Show loading state on button
    showLoadingState(btn) {
        btn.classList.add('loading');
        btn.innerHTML = '<span class="loading-spinner"></span>Analyzing Flight...';
    }

    // Reset button to normal state
    resetLoadingState(btn) {
        btn.classList.remove('loading');
        btn.innerHTML = '<i class="fas fa-magic"></i> Predict Flight Delay';
    }

    // Client-side heuristic predictor (mirrors demo model logic)
    predictClientSide() {
        const formData = new FormData(document.getElementById('predictionForm'));
        const origin = (formData.get('origin') || '').toUpperCase();
        const destination = (formData.get('destination') || '').toUpperCase();
        const dateStr = formData.get('flightDate');
        const depTime = formData.get('departureTime');

        // Weather factors
        const tempC = Number(formData.get('temperature')) || 0; // °C
        const precipMm = Number(formData.get('precipitation')) || 0; // mm
        const windKmh = Number(formData.get('windSpeed')) || 0; // km/h (UI)
        const windMps = windKmh / 3.6; // convert to m/s to mirror backend logic

        let delay = 5.0; // base

        // Peak-hour contribution from departure time
        if (depTime) {
            const hour = parseInt(depTime.split(':')[0], 10);
            if ((hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 18)) {
                delay += 10.0;
            }
        }

        // Weather impacts similar to backend demo
        if (Math.abs(tempC - 20) > 15) delay += 5.0;
        if (precipMm > 5) delay += 8.0;
        if (windMps > 10) delay += 7.0;

        // Hub/route heuristic
        const hubs = ['ORD','DEN','IAH','EWR','SFO','LAX','IAD'];
        if (hubs.includes(origin) || hubs.includes(destination)) delay += 3.0;
        if (hubs.includes(origin) && hubs.includes(destination)) delay += 2.0;

        // January constraint check (UI already enforces; double-check)
        if (dateStr) {
            const d = new Date(dateStr);
            if (!(d.getUTCFullYear() === 2025 && d.getUTCMonth() === 0)) {
                // Not January 2025
                delay += 5.0;
            }
        }

        // Clamp and round
        delay = Math.max(0, Math.round(delay));
        const confidence = 85; // fixed to mirror backend demo

        return {
            delay,
            confidence,
            status: this.getDelayStatus(delay),
            recommendation: this.getRecommendation(delay),
            route: `${origin} → ${destination}`
        };
    }

    // Server-side prediction via API_URL
    async predictServerSide() {
        const formData = new FormData(document.getElementById('predictionForm'));

        const origin = (formData.get('origin') || '').toUpperCase();
        const destination = (formData.get('destination') || '').toUpperCase();
        const dateStr = formData.get('flightDate');
        const depTime = formData.get('departureTime');
        const arrTime = formData.get('arrivalTime');
        const tempC = Number(formData.get('temperature')) || 0;
        const precipMm = Number(formData.get('precipitation')) || 0;
        const windKmh = Number(formData.get('windSpeed')) || 0;
        const windMps = windKmh / 3.6;

        const toMinutes = (hhmm) => {
            const [h, m] = (hhmm || '00:00').split(':').map(Number);
            return (h * 60) + m;
        };

        const body = {
            Origin: origin,
            Dest: destination,
            FlightDate: dateStr,
            CRSDepTime: toMinutes(depTime),
            CRSArrTime: toMinutes(arrTime),
            temperature_c: tempC,
            precip_mm: precipMm,
            cloud_pct: 0,
            wind_speed_mps: windMps
        };

        const resp = await fetch(`${API_URL.replace(/\/$/, '')}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!resp.ok) throw new Error(`API ${resp.status}`);
        const data = await resp.json();
        if (data.status !== 'success') throw new Error(data.error || 'Prediction failed');

        const delay = Math.max(0, Math.round(Number(data.predicted_delay) || 0));
        const confidence = Math.round((Number(data.confidence) || 0.85) * 100);
        return {
            delay,
            confidence,
            status: this.getDelayStatus(delay),
            recommendation: this.getRecommendation(delay),
            route: `${origin} → ${destination}`
        };
    }

    // Determine delay status based on minutes
    getDelayStatus(delay) {
        if (delay <= 5) {
            return { 
                text: 'On Time', 
                class: 'status-success', 
                icon: 'fas fa-check-circle' 
            };
        } else if (delay <= 15) {
            return { 
                text: 'Minor Delay', 
                class: 'status-warning', 
                icon: 'fas fa-exclamation-triangle' 
            };
        } else if (delay <= 30) {
            return { 
                text: 'Moderate Delay', 
                class: 'status-warning', 
                icon: 'fas fa-clock' 
            };
        } else {
            return { 
                text: 'Significant Delay', 
                class: 'status-danger', 
                icon: 'fas fa-times-circle' 
            };
        }
    }

    // Get recommendation based on delay
    getRecommendation(delay) {
        if (delay <= 5) {
            return "✅ Your flight is likely to be on time! Arrive at the airport 2 hours before departure for international flights, 1 hour for domestic flights.";
        } else if (delay <= 15) {
            return "⚠️ Minor delays expected. Consider arriving 2.5 hours early and check for real-time updates from your airline.";
        } else if (delay <= 30) {
            return "⚠️ Moderate delays likely. Plan for extra time, check alternative flights, and consider airport amenities for a longer wait.";
        } else {
            return "🚨 Significant delays expected. Strongly consider rebooking, check alternative routes, or prepare for extended wait times.";
        }
    }

    // Display prediction results
    displayResults(prediction) {
        const elements = {
            delayTime: document.getElementById('delayTime'),
            delayStatus: document.getElementById('delayStatus'),
            delayIndicator: document.getElementById('delayIndicator'),
            delayIcon: document.getElementById('delayIcon'),
            confidenceScore: document.getElementById('confidenceScore'),
            recommendationText: document.getElementById('recommendationText')
        };
        
        // Update content
        elements.delayTime.textContent = prediction.delay + ' min';
        elements.delayStatus.textContent = prediction.status.text;
        elements.confidenceScore.textContent = prediction.confidence;
        elements.recommendationText.textContent = prediction.recommendation;
        
        // Update styling
        elements.delayIndicator.className = 'delay-indicator ' + prediction.status.class;
        elements.delayIcon.className = prediction.status.icon;
        
        // Add pulse animation to delay time
        this.addPulseAnimation(elements.delayTime);
        
        // Log prediction for debugging
        console.log('Flight AI Prediction:', prediction);
    }

    // Add pulse animation to element
    addPulseAnimation(element) {
        element.style.animation = 'none';
        element.offsetHeight; // Trigger reflow
        element.style.animation = 'pulse 2s ease-in-out infinite';
    }

    // Utility method to format time
    formatTime(timeString) {
        const [hours, minutes] = timeString.split(':');
        const hour = parseInt(hours);
        const ampm = hour >= 12 ? 'PM' : 'AM';
        const hour12 = hour % 12 || 12;
        return `${hour12}:${minutes} ${ampm}`;
    }

    // Get current weather (placeholder for actual weather API)
    async getCurrentWeather(airportCode) {
        // In production, integrate with a weather API
        // This is a placeholder that returns mock data
        return {
            temperature: 20 + Math.random() * 20,
            precipitation: Math.random() * 10,
            windSpeed: Math.random() * 30
        };
    }

    // Analytics tracking (placeholder)
    trackEvent(eventName, properties = {}) {
        console.log('Analytics Event:', eventName, properties);
        // In production, integrate with analytics service
    }
}

// Global utility functions
window.updateTempDisplay = function(value) {
    flightAI.updateTempDisplay(value);
};

window.updatePrecipDisplay = function(value) {
    flightAI.updatePrecipDisplay(value);
};

window.updateWindDisplay = function(value) {
    flightAI.updateWindDisplay(value);
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.flightAI = new FlightAI();
    console.log('✈️ Flight AI System Initialized');
});

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('SW registered: ', registration);
            })
            .catch(function(registrationError) {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
