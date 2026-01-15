// FILEPATH: src/soen/utils/physical_mappings/static/js/script.js

// Format number for display with appropriate scientific notation
function formatValue(value) {
    if (value === undefined || value === null) return '';
    if (value === Infinity) return '\\infty';
    if (value === -Infinity) return '-\\infty';
    if (isNaN(value)) return 'undefined';
    
    // Convert to scientific notation with 4 significant figures
    const str = value.toExponential(4);
    let [coef, exp] = str.split('e');
    exp = parseInt(exp);

    // If exponent is zero, show a plain number (no ×10^0)
    if (exp === 0) {
        // Use a compact format that preserves up to 6 meaningful digits
        const plain = Number(value).toPrecision(6);
        return plain;
    }

    // Format coefficients near 1 more nicely
    if (Math.abs(parseFloat(coef)) === 1) {
        coef = coef[0] === '-' ? '-1' : '1';
    }
    
    return `${coef} \\times 10^{${exp}}`;
}

// Format full equation with value for display
function formatEquation(latex, value, unit = '') {
    const formattedValue = formatValue(value);
    if (unit) {
        return `$${latex} = ${formattedValue}\\,\\text{${unit}}$`;
    }
    return `$${latex} = ${formattedValue}$`;
}

// Safely request MathJax to typeset if available (v3 only)
function safeTypeset() {
    try {
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            return window.MathJax.typesetPromise();
        }
    } catch (e) {
        // ignore and fall through
    }
    // MathJax not available (e.g., offline); avoid throwing and continue
    return Promise.resolve();
}

// Show error message in a result div (defensive when element missing)
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="error">${message}</div>`;
    } else {
        console.error(`showError target not found: #${elementId}`, message);
    }
}

// Simple debounce to avoid excessive server calls on input
function debounce(fn, delay = 300) {
    let timerId;
    return (...args) => {
        if (timerId) clearTimeout(timerId);
        timerId = setTimeout(() => fn.apply(null, args), delay);
    };
}



async function updateBaseParameters() {
    const button = document.querySelector('.base-parameters button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            I_c: parseFloat(document.getElementById('I_c').value),
            gamma_c: parseFloat(document.getElementById('gamma_c').value),
            beta_c: parseFloat(document.getElementById('beta_c').value)
        };

        // Validate inputs
        if (Object.values(data).some(val => isNaN(val) || val <= 0)) {
            throw new Error('All parameters must be positive numbers');
        }

        const response = await fetch('/update_base_parameters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const result = await response.json();

        // Update all parameter displays with LaTeX formatting
        for (const [key, data] of Object.entries(result)) {
            const element = document.getElementById(key);
            if (element) {
                let unit = '';
                // Add appropriate units
                switch(key) {
                    case 'I_c': unit = 'A'; break;
                    case 'gamma_c': unit = 'F/A'; break;
                    case 'c_j': unit = 'F'; break;
                    case 'r_jj': unit = '\u03A9'; break;
                    case 'V_j': unit = 'V'; break;
                    case 'tau_0': unit = 's'; break;
                    case 'omega_c':
                    case 'omega_p': unit = 'rad/s'; break;
                }
                element.innerHTML = formatEquation(data.latex, data.value, unit);
            }
        }

        // After updating base parameters, also update any derived parameters
        await updateDerivedParameters();

        // Trigger MathJax to rerender all equations (if available)
        safeTypeset();

    } catch (error) {
        showError('baseParametersResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}

// Convert physical values to dimensionless
async function convertToDimensionless() {
    const button = document.querySelector('.conversion-column:first-child button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            I: parseFloat(document.getElementById('I').value),
            Phi: parseFloat(document.getElementById('Phi').value),
            L: parseFloat(document.getElementById('L').value),
            t: parseFloat(document.getElementById('t').value),
            r_leak: parseFloat(document.getElementById('r_leak').value),
            G_fq: parseFloat(document.getElementById('G_fq').value),
            tau_physical: parseFloat(document.getElementById('tau_physical').value)
        };

        // Remove NaN values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        // Validate we have at least one value
        if (Object.keys(data).length === 0) {
            throw new Error('Please enter at least one value to convert');
        }

        const response = await fetch('/convert_to_dimensionless', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Format and display results
        const resultHtml = Object.entries(result)
            .map(([key, data]) => formatEquation(data.latex, data.value))
            .join('<br>');
        
        document.getElementById('dimensionlessResult').innerHTML = resultHtml;

        // Calculate derived parameters if we have the necessary values
        if (result.beta_L && result.alpha) {
            calculateDerivedParameters(result.beta_L.value, result.alpha.value);
        }

        // Trigger MathJax to rerender equations (if available)
        safeTypeset();

    } catch (error) {
        showError('dimensionlessResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}

// Convert dimensionless values to physical
async function convertToPhysical() {
    const button = document.querySelector('.conversion-column:last-child button');
    button.disabled = true;
    
    try {
        // Collect input values
        const data = {
            i: parseFloat(document.getElementById('i').value),
            phi: parseFloat(document.getElementById('phi').value),
            beta_L: parseFloat(document.getElementById('beta_L').value),
            gamma_plus: parseFloat(document.getElementById('gamma_plus').value),
            gamma_minus: parseFloat(document.getElementById('gamma_minus').value),
            t_prime: parseFloat(document.getElementById('t_prime').value),
            alpha: parseFloat(document.getElementById('alpha').value),
            g_fq: parseFloat(document.getElementById('g_fq').value)
        };

        // Remove NaN values
        Object.keys(data).forEach(key => {
            if (isNaN(data[key])) delete data[key];
        });

        // Validate we have at least one value
        if (Object.keys(data).length === 0) {
            throw new Error('Please enter at least one value to convert');
        }

        const response = await fetch('/convert_to_physical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Format and display results
        const resultHtml = Object.entries(result)
            .map(([key, data]) => formatEquation(data.latex, data.value, data.unit))
            .join('<br>');
        
        document.getElementById('physicalResult').innerHTML = resultHtml;
        
        // If we have alpha but not r_leak in the results, calculate and display r_leak
        if (data.alpha && !result.r_leak) {
            try {
                // Fetch r_jj from the current base parameters
                const r_jj_element = document.getElementById('r_jj');
                if (r_jj_element) {
                    // Make a call to convert alpha to r_leak
                    fetch('/convert_to_physical', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ alpha: data.alpha })
                    }).then(response => response.json())
                    .then(r_leak_result => {
                        if (r_leak_result.r_leak) {
                            // Append r_leak to the results
                            const currentHtml = document.getElementById('physicalResult').innerHTML;
                            document.getElementById('physicalResult').innerHTML = 
                                currentHtml + '<br>' + formatEquation(r_leak_result.r_leak.latex, r_leak_result.r_leak.value, r_leak_result.r_leak.unit);
                            safeTypeset();
                        }
                    });
                }
            } catch (error) {
                console.error("Error calculating r_leak from alpha:", error);
            }
        }

        // Calculate derived parameters if we have beta_L and alpha
        if (data.beta_L && data.alpha) {
            calculateDerivedParameters(data.beta_L, data.alpha);
        }

        // Trigger MathJax to rerender equations (if available)
        safeTypeset();

    } catch (error) {
        showError('physicalResult', `Error: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}


// Add event listeners for parameters that can trigger updates
document.getElementById('beta_L').addEventListener('input', updateDerivedParameters);
document.getElementById('alpha').addEventListener('input', updateDerivedParameters);
document.getElementById('gamma_plus').addEventListener('input', updateDerivedParameters);
document.getElementById('gamma_minus').addEventListener('input', updateDerivedParameters);

// Function to update derived parameters based on user input
async function updateDerivedParameters() {
    // Get all parameter values
    const beta_L = parseFloat(document.getElementById('beta_L').value);
    const gamma_plus = parseFloat(document.getElementById('gamma_plus').value);
    const alpha = parseFloat(document.getElementById('alpha').value);
    const gamma_minus = parseFloat(document.getElementById('gamma_minus').value);
    
    // Prepare data for API call
    const data = {};
    
    if (!isNaN(beta_L)) {
        data.beta_L = beta_L;
    }
    
    if (!isNaN(gamma_plus)) {
        data.gamma_plus = gamma_plus;
    }
    
    if (!isNaN(alpha)) {
        data.alpha = alpha;
    }
    
    if (!isNaN(gamma_minus)) {
        data.gamma_minus = gamma_minus;
    }
    
    // Only call API if we have at least one parameter
    if (Object.keys(data).length > 0) {
        await calculateDerivedParameters(data);
    }
}

// Calculate and update derived dimensionless parameters
async function calculateDerivedParameters(data) {
    try {
        const response = await fetch('/calculate_derived', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }

        const result = await response.json();

        // Update UI with calculated values without overwriting user inputs
        updateUIWithCalculatedValues(result);
        safeTypeset();

    } catch (error) {
        console.error('Error calculating derived parameters:', error);
    }
}

// Update UI with calculated values without overwriting user input
function updateUIWithCalculatedValues(result) {
    // Update tau display
    if (result.tau) {
        document.getElementById('tau').innerHTML = formatEquation(result.tau.latex, result.tau.value);
    }
    
    // Update other fields only if they're not currently being edited
    const activeElement = document.activeElement;
    
    if (result.beta_L && activeElement.id !== 'beta_L' && document.getElementById('beta_L').value === '') {
        document.getElementById('beta_L').value = result.beta_L.value;
    }
    
    if (result.gamma_plus && activeElement.id !== 'gamma_plus' && document.getElementById('gamma_plus').value === '') {
        document.getElementById('gamma_plus').value = result.gamma_plus.value;
    }
    
    if (result.alpha && activeElement.id !== 'alpha' && document.getElementById('alpha').value === '') {
        document.getElementById('alpha').value = result.alpha.value;
    }
    
    if (result.gamma_minus && activeElement.id !== 'gamma_minus' && document.getElementById('gamma_minus').value === '') {
        document.getElementById('gamma_minus').value = result.gamma_minus.value;
    }
    
    // Display r_leak in the physicalResult div if it's calculated
    if (result.r_leak) {
        // First check if there's an existing result div
        let physicalResult = document.getElementById('physicalResult');
        if (!physicalResult || physicalResult.innerHTML.trim() === '') {
            physicalResult = document.getElementById('physicalResult');
            physicalResult.innerHTML = formatEquation(result.r_leak.latex, result.r_leak.value, result.r_leak.unit);
        } else {
            // Append to existing results
            const resultHtml = physicalResult.innerHTML;
            if (!resultHtml.includes('r_{\\text{leak}}')) {
                physicalResult.innerHTML = resultHtml + '<br>' + formatEquation(result.r_leak.latex, result.r_leak.value, result.r_leak.unit);
            }
        }
        // Trigger MathJax to render (if available)
        safeTypeset();
    }
}

// Link gamma_plus and beta_L inputs
document.getElementById('gamma_plus').addEventListener('input', function(e) {
    const gamma_plus = parseFloat(e.target.value);
    if (!isNaN(gamma_plus) && gamma_plus !== 0) {
        document.getElementById('beta_L').value = 1 / gamma_plus;
        // Trigger update of other derived parameters
        updateDerivedParameters();
    }
});

document.getElementById('beta_L').addEventListener('input', function(e) {
    const beta_L = parseFloat(e.target.value);
    if (!isNaN(beta_L) && beta_L !== 0) {
        document.getElementById('gamma_plus').value = 1 / beta_L;
        // Trigger update of other derived parameters
        updateDerivedParameters();
    }
});

// Load and display fundamental constants
async function loadConstants() {
    try {
        const response = await fetch('/get_constants');
        if (!response.ok) {
            throw new Error('Failed to fetch constants');
        }
        
        const constants = await response.json();
        const constantsContainer = document.getElementById('constants-container');
        
        // Clear existing content
        constantsContainer.innerHTML = '';
        
        // Create a card for each constant
        for (const [key, data] of Object.entries(constants)) {
            const card = document.createElement('div');
            card.className = 'constant-card';
            
            const nameElement = document.createElement('div');
            nameElement.className = 'constant-name';
            nameElement.innerHTML = `$${data.latex}$ - ${data.description}`;
            
            const valueElement = document.createElement('div');
            valueElement.className = 'constant-value';
            valueElement.innerHTML = `${formatEquation(data.latex, data.value, data.unit)}`;
            
            card.appendChild(nameElement);
            card.appendChild(valueElement);
            
            constantsContainer.appendChild(card);
        }
        
        // Also apply centralized defaults to the base-parameter inputs before computing derived values
        try {
            const icDefault = constants?.I_c_default?.value;
            const gammaCDefault = constants?.gamma_c_default?.value;
            const betaCDefault = constants?.beta_c_default?.value;

            const icInput = document.getElementById('I_c');
            const gammaCInput = document.getElementById('gamma_c');
            const betaCInput = document.getElementById('beta_c');

            if (icInput && typeof icDefault === 'number') {
                icInput.value = icDefault;
            }
            if (gammaCInput && typeof gammaCDefault === 'number') {
                gammaCInput.value = gammaCDefault;
            }
            if (betaCInput && typeof betaCDefault === 'number') {
                betaCInput.value = betaCDefault;
            }

            // Now compute and display all derived values using these defaults
            await updateBaseParameters();
        } catch (e) {
            // If anything goes wrong applying defaults, still attempt to compute derived values
            await updateBaseParameters();
        }

        // Render Math expressions (if available)
        safeTypeset();
        
    } catch (error) {
        console.error('Error loading constants:', error);
        const constantsContainer = document.getElementById('constants-container');
        constantsContainer.innerHTML = `<div class="error">Error loading constants: ${error.message}</div>`;
        // Attempt to compute derived values even if constants failed to load
        try { await updateBaseParameters(); } catch (_) {}
    }
}

// Copy to clipboard functionality
function addCopyToClipboardButtons() {
    // Add copy buttons to all value displays
    document.querySelectorAll('.constant-value, .parameter-item span:nth-child(2), .result').forEach(el => {
        // Only add if it contains a value and doesn't already have a copy button
        if (el.textContent.trim() && !el.querySelector('.copy-btn')) {
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = '<span class="copy-icon">📋</span>';
            copyBtn.title = 'Copy value to clipboard';
            
            copyBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                
                // Clone the parent element `el` to get its text content without the button itself.
                const elClone = el.cloneNode(true);
                // Find and remove the copy button (which has the class 'copy-btn') from the clone.
                const buttonInClone = elClone.querySelector('.copy-btn');
                if (buttonInClone) {
                    elClone.removeChild(buttonInClone);
                }
                
                // Use the clone's content for text extraction
                const originalTextFromClone = elClone.innerHTML;
                const textContentFromClone = elClone.textContent;

                let scientificNotation;
                
                // Look for the pattern: digits × 10^{exponent} from the clone's innerHTML
                const match = originalTextFromClone.match(/(\d+\.\d+|\d+)\s*\\times\s*10\^\{([+-]?\d+)\}/);
                if (match) {
                    const coefficient = parseFloat(match[1]);
                    const exponent = parseInt(match[2]);
                    // Format as standard scientific notation (e.g., 2.067e-15)
                    scientificNotation = `${coefficient}e${exponent}`;
                } else {
                    // Fallback to plain text extraction from the clone's textContent
                    scientificNotation = textContentFromClone.replace(/.*=\s*/, '').trim();
                }
                
                navigator.clipboard.writeText(scientificNotation).then(() => {
                    copyBtn.classList.add('copied');
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                    }, 1500);
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                });
            });
            
            el.appendChild(copyBtn);
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadConstants();
    
    // Add copy buttons after a short delay to ensure all elements are rendered
    setTimeout(addCopyToClipboardButtons, 1000);
    
    // Add event listener for parameter changes to update copy buttons
    const observer = new MutationObserver(function(mutations) {
        addCopyToClipboardButtons();
    });
    
    // Observe the entire container for changes
    observer.observe(document.querySelector('.container'), { 
        childList: true, 
        subtree: true, 
        characterData: true
    });

    // Auto-update derived base parameters when inputs change
    const debouncedUpdateBase = debounce(updateBaseParameters, 300);
    const icEl = document.getElementById('I_c');
    const gcEl = document.getElementById('gamma_c');
    const bcEl = document.getElementById('beta_c');
    if (icEl) icEl.addEventListener('input', debouncedUpdateBase);
    if (gcEl) gcEl.addEventListener('input', debouncedUpdateBase);
    if (bcEl) bcEl.addEventListener('input', debouncedUpdateBase);
});
