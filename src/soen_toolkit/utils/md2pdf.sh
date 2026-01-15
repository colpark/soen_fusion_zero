#!/usr/bin/env bash

#===============================================================================
# md2pdf.sh - Convert Markdown to PDF with professional styling
#===============================================================================
# Version: 2.1.1
# Features:
#   - Professional GitHub-inspired styling
#   - Enhanced math equation support (MathJax)
#   - Improved table overflow handling
#   - Image handling (both Markdown and HTML syntax)
#   - Cross-platform (macOS/Linux)
#   - Multi-engine comparison mode
#   - High-quality output
#   - Smart engine recommendation
#===============================================================================

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Temporary directory for intermediate files
TEMP_DIR=""

#===============================================================================
# Helper Functions
#===============================================================================

print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] INPUT.md [OUTPUT.pdf]

Convert Markdown to PDF with professional styling and math support.

Arguments:
  INPUT.md          Input Markdown file
  OUTPUT.pdf        Output PDF file (optional, defaults to INPUT.pdf)

Options:
  -h, --help        Show this help message
  -v, --verbose     Enable verbose output
  -k, --keep-temp   Keep temporary files for debugging
  -a, --all         Generate PDFs with ALL available engines for comparison

Examples:
  $(basename "$0") README.md
  $(basename "$0") document.md output.pdf
  $(basename "$0") -a document.md    # Compare all engines
  $(basename "$0") -v notes.md
EOF
}

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        if [[ "${KEEP_TEMP:-0}" -eq 0 ]]; then
            rm -rf "$TEMP_DIR"
        else
            print_info "Temporary files kept in: $TEMP_DIR"
        fi
    fi
}

trap cleanup EXIT

#===============================================================================
# Dependency Checking
#===============================================================================

check_command() {
    command -v "$1" &> /dev/null
}

check_chrome_macos() {
    # Check for Chrome.app on macOS
    [[ -d "/Applications/Google Chrome.app" ]]
}

check_dependencies() {
    local missing_deps=()
    
    if ! check_command pandoc; then
        missing_deps+=("pandoc")
    fi
    
    # Check for at least one PDF engine (prioritize Chrome for best math rendering)
    local has_pdf_engine=0
    if check_command google-chrome || check_command chromium || check_command chromium-browser || check_chrome_macos; then
        has_pdf_engine=1
        if check_command google-chrome; then
            PDF_ENGINE="chrome"
        elif check_chrome_macos; then
            PDF_ENGINE="chrome-macos"
        elif check_command chromium; then
            PDF_ENGINE="chromium"
        else
            PDF_ENGINE="chromium-browser"
        fi
    elif check_command weasyprint; then
        has_pdf_engine=1
        PDF_ENGINE="weasyprint"
    elif check_command xelatex || check_command pdflatex; then
        has_pdf_engine=1
        PDF_ENGINE=$(check_command xelatex && echo "xelatex" || echo "pdflatex")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        echo "Installation instructions:"
        echo ""
        echo "macOS:    brew install pandoc"
        echo "Ubuntu:   sudo apt-get install pandoc"
        echo "Fedora:   sudo dnf install pandoc"
        echo ""
        return 1
    fi
    
    if [[ $has_pdf_engine -eq 0 ]]; then
        print_error "No PDF conversion engine found"
        echo ""
        echo "Install one of: weasyprint (pip), Chrome, or LaTeX"
        return 1
    fi
    
    return 0
}

#===============================================================================
# Engine Recommendation
#===============================================================================

recommend_engine() {
    local input_md="$1"
    
    # Check for math content
    if grep -qE '\$.*\$|\\begin\{.*\}|\\end\{.*\}' "$input_md"; then
        if check_command google-chrome || check_command chromium || check_command chromium-browser; then
            print_info "Document contains math - Chrome/Chromium recommended for best rendering"
            return 0
        else
            print_warning "Document contains math but Chrome not available"
            print_warning "Consider installing Chrome for better math rendering"
            print_warning "  macOS:  brew install --cask google-chrome"
            print_warning "  Ubuntu: sudo apt-get install chromium-browser"
        fi
    fi
    
    # Check for large tables
    local table_count=$(grep -c '^\|' "$input_md" || true)
    if [[ $table_count -gt 5 ]]; then
        print_info "Document has $table_count tables - verify table formatting in output"
    fi
}

#===============================================================================
# Professional CSS Styling
#===============================================================================

create_professional_css() {
    local css_file="$1"
    
    cat > "$css_file" << 'EOFCSS'
@page {
    margin: 0.75in;
    size: letter;
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    font-size: 10pt;
    line-height: 1.6;
    color: #24292f;
    background: white;
    margin: 0;
    padding: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
    line-height: 1.25;
    color: #1f2328;
    page-break-after: avoid;
}

h1 {
    font-size: 2em;
    font-weight: 700;
    border-bottom: 2px solid #d8dee4;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 24px;
}

h2 {
    font-size: 1.5em;
    border-bottom: 1px solid #d8dee4;
    padding-bottom: 0.3em;
    margin-top: 32px;
}

h3 {
    font-size: 1.25em;
    margin-top: 24px;
}

h4 {
    font-size: 1em;
    margin-top: 20px;
}

h5 {
    font-size: 0.875em;
    margin-top: 16px;
}

h6 {
    font-size: 0.85em;
    color: #656d76;
    margin-top: 16px;
}

/* Paragraphs */
p {
    margin-top: 0;
    margin-bottom: 16px;
}

/* Links */
a {
    color: #0969da;
    text-decoration: none;
    font-weight: 500;
}

a:hover {
    text-decoration: underline;
}

/* Strong/Bold */
strong, b {
    font-weight: 600;
    color: #1f2328;
}

/* Emphasis/Italic */
em, i {
    font-style: italic;
}

/* Inline Code */
code {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.875em;
    padding: 0.2em 0.4em;
    margin: 0;
    background-color: rgba(175, 184, 193, 0.2);
    border-radius: 6px;
}

/* Code Blocks */
pre {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.85em;
    line-height: 1.45;
    background-color: #f6f8fa;
    border-radius: 6px;
    padding: 16px;
    overflow: auto;
    margin: 16px 0;
    border: 1px solid #d0d7de;
    page-break-inside: avoid;
}

pre code {
    font-size: 100%;
    padding: 0;
    margin: 0;
    background-color: transparent;
    border: 0;
    border-radius: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Blockquotes */
blockquote {
    margin: 0 0 16px 0;
    padding: 0 1em;
    color: #656d76;
    border-left: 0.25em solid #d0d7de;
}

/* Lists */
ul, ol {
    margin-top: 0;
    margin-bottom: 16px;
    padding-left: 2em;
}

li {
    margin-top: 0.25em;
}

li + li {
    margin-top: 0.25em;
}

/* Tables - Enhanced for overflow handling */
table {
    border-spacing: 0;
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    overflow: auto;
    margin: 16px 0;
    page-break-inside: avoid;
    word-wrap: break-word;
}

table th {
    font-weight: 600;
    padding: 6px 13px;
    border: 1px solid #d0d7de;
    background-color: #f6f8fa;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}

table td {
    padding: 6px 13px;
    border: 1px solid #d0d7de;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
    max-width: 200px;
}

table tr {
    background-color: #ffffff;
    border-top: 1px solid hsla(210, 18%, 87%, 1);
}

table tr:nth-child(2n) {
    background-color: #f6f8fa;
}

/* Shrink tables slightly for print if needed */
@media print {
    table {
        font-size: 0.85em;
    }
}

/* Images */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto;
    border-radius: 6px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    page-break-inside: avoid;
}

/* Horizontal Rules */
hr {
    height: 0.25em;
    padding: 0;
    margin: 24px 0;
    background-color: #d8dee4;
    border: 0;
}

/* Syntax Highlighting */
.sourceCode { background-color: transparent; }
.kw { color: #d73a49; font-weight: bold; }
.dt { color: #005cc5; }
.dv, .bn, .fl { color: #005cc5; }
.ch, .st { color: #032f62; }
.co { color: #6a737d; font-style: italic; }
.ot { color: #6f42c1; }
.al { color: #e36209; }
.fu { color: #6f42c1; font-weight: 600; }
.er { color: #d73a49; font-weight: bold; }
.wa { color: #6a737d; }
.cn { color: #005cc5; }
.sc, .vs, .ss { color: #032f62; }
.im { color: #d73a49; }
.va { color: #e36209; }
.cf { color: #d73a49; font-weight: bold; }
.op { color: #005cc5; }
.bu { color: #005cc5; }
.pp { color: #bc4c00; font-weight: bold; }

/* Math - Enhanced styling */
.math {
    overflow-x: auto;
    margin: 16px 0;
}

.math.inline {
    display: inline;
    margin: 0 0.1em;
    vertical-align: baseline;
}

.math.display {
    display: block;
    margin: 1em 0;
    text-align: center;
    overflow-x: auto;
}

/* MathJax container styling */
mjx-container {
    overflow-x: auto;
    overflow-y: hidden;
}

mjx-container[display="true"] {
    max-width: 100%;
    overflow-x: auto;
    margin: 1em 0;
}

.MathJax {
    font-size: 1em !important;
}

/* Page breaks */
h1, h2, h3, h4, h5, h6 {
    page-break-after: avoid;
}

pre, table, figure {
    page-break-inside: avoid;
}
EOFCSS
}

#===============================================================================
# Image Processing
#===============================================================================

process_images() {
    local input_md="$1"
    local output_md="$2"
    local input_dir
    input_dir="$(cd "$(dirname "$input_md")" && pwd)"
    
    [[ -n "${VERBOSE:-}" ]] && print_info "Processing images from: $input_dir"
    
    # Read the markdown file
    cp "$input_md" "$output_md"
    
    # Process Markdown images: ![alt](path)
    # Use grep to find lines with markdown images, then process them
    (grep -n '!\[.*\](.*' "$input_md" || true) | while IFS=: read -r line_num line_content; do
        # Extract image paths using sed
        echo "$line_content" | grep -oE '!\[[^]]*\]\([^)]+\)' | while read -r img_match; do
            local img_path=$(echo "$img_match" | sed -n 's/.*(\([^)]*\)).*/\1/p')
            
            # Skip URLs
            [[ $img_path =~ ^https?:// ]] && continue
            [[ $img_path =~ ^data: ]] && continue
            [[ $img_path =~ ^file:// ]] && continue
            
            # Convert relative to absolute
            if [[ ! $img_path =~ ^/ ]]; then
                local abs_path="$input_dir/$img_path"
                if [[ -f "$abs_path" ]]; then
                    local file_url="file://$abs_path"
                    # Escape special characters for sed
                    local escaped_path=$(printf '%s\n' "$img_path" | sed 's/[\[\.*^$/]/\\&/g')
                    sed -i.bak "s|]($escaped_path)|]($file_url)|g" "$output_md"
                    [[ -n "${VERBOSE:-}" ]] && print_info "Markdown image: $img_path -> $file_url"
                else
                    print_warning "Image not found: $img_path"
                fi
            fi
        done
    done
    
    # Process HTML images: <img src="path">
    (grep -oE '<img[^>]+>' "$output_md" || true) | while read -r img_tag; do
        # Extract src attribute value - handle both single and double quotes
        local img_path=""
        
        # Try double quotes first
        if [[ $img_tag =~ src=\"([^\"]+)\" ]]; then
            img_path="${BASH_REMATCH[1]}"
        # Try single quotes
        elif [[ $img_tag =~ src=\'([^\']+)\' ]]; then
            img_path="${BASH_REMATCH[1]}"
        # Try without quotes (technically invalid HTML but sometimes happens)
        elif [[ $img_tag =~ src=([^\ \>]+) ]]; then
            img_path="${BASH_REMATCH[1]}"
        fi
        
        [[ -z "$img_path" ]] && continue
        [[ $img_path =~ ^https?:// ]] && continue
        [[ $img_path =~ ^data: ]] && continue
        [[ $img_path =~ ^file:// ]] && continue
        
        if [[ ! $img_path =~ ^/ ]]; then
            local abs_path="$input_dir/$img_path"
            if [[ -f "$abs_path" ]]; then
                local file_url="file://$abs_path"
                # Escape special characters for sed - use @ as delimiter to avoid issues with /
                local escaped_path=$(printf '%s\n' "$img_path" | sed 's/[@&]/\\&/g')
                local escaped_url=$(printf '%s\n' "$file_url" | sed 's/[@&]/\\&/g')
                sed -i.bak "s@src=[\"']${escaped_path}[\"']@src=\"${escaped_url}\"@g" "$output_md"
                [[ -n "${VERBOSE:-}" ]] && print_info "HTML image: $img_path -> $file_url"
            else
                print_warning "HTML image not found: $img_path (looking at: $abs_path)"
            fi
        fi
    done
    
    rm -f "$output_md.bak"
}

#===============================================================================
# Conversion Functions
#===============================================================================

convert_with_weasyprint() {
    local input_md="$1"
    local output_pdf="$2"
    local css_file="$3"
    
    local html_file="$TEMP_DIR/weasyprint.html"
    
    print_info "Generating HTML with high-resolution equation images..."
    
    # WeasyPrint doesn't support JavaScript, so use webtex with high DPI
    # Using codecogs with 300 DPI for much better quality
    # Disable implicit_figures to prevent alt text from becoming captions
    pandoc "$input_md" \
        -f markdown+tex_math_dollars-implicit_figures \
        -t html5 \
        --standalone \
        --webtex='https://latex.codecogs.com/png.latex?\dpi{300}' \
        --css="$css_file" \
        --syntax-highlighting=pygments \
        --embed-resources \
        ${VERBOSE:+--verbose} \
        -o "$html_file"
    
    print_info "Converting to PDF with WeasyPrint..."
    
    if [[ -n "${VERBOSE:-}" ]]; then
        weasyprint "$html_file" "$output_pdf"
    else
        weasyprint "$html_file" "$output_pdf" 2>/dev/null
    fi
}

convert_with_chrome() {
    local input_md="$1"
    local output_pdf="$2"
    local css_file="$3"
    
    local html_file="$TEMP_DIR/chrome.html"
    
    print_info "Generating HTML with MathJax..."
    
    # Generate HTML with MathJax CDN and embed local resources (images)
    pandoc "$input_md" \
        -f markdown+tex_math_dollars \
        -t html5 \
        --standalone \
        --mathjax='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' \
        --css="$css_file" \
        --syntax-highlighting=pygments \
        --embed-resources \
        --metadata=pagetitle:"Document" \
        ${VERBOSE:+--verbose} \
        -o "$html_file"
    
    # Add custom MathJax configuration for better rendering
    if grep -q 'mathjax' "$html_file"; then
        # Create a temporary file with the MathJax config
        local mathjax_config_file="${html_file}.mathjax.tmp"
        cat > "$mathjax_config_file" << 'EOF'
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      ignoreHtmlClass: "tex2jax_ignore"
    },
    svg: {
      fontCache: "global",
      scale: 1
    },
    startup: {
      ready: function() {
        MathJax.startup.defaultReady();
        MathJax.startup.promise.then(function() {
          console.log('MathJax rendering complete');
          document.body.classList.add('mathjax-ready');
        });
      }
    }
  };
</script>
EOF
        
        # Find the line with mathjax and insert config before it
        local temp_file="${html_file}.tmp"
        awk '/mathjax/ {
            system("cat '"$mathjax_config_file"'")
        }
        { print }
        ' "$html_file" > "$temp_file"
        mv "$temp_file" "$html_file"
        rm -f "$mathjax_config_file"
    fi
    
    print_info "Converting to PDF with Chrome..."
    print_info "Allowing time for MathJax to render equations..."
    
    local chrome_bin=""
    case "$PDF_ENGINE" in
        chrome) chrome_bin="google-chrome" ;;
        chrome-macos) chrome_bin="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ;;
        chromium) chrome_bin="chromium" ;;
        chromium-browser) chrome_bin="chromium-browser" ;;
    esac
    
    # Use timeout to give MathJax time to load and render from CDN
    # Start Chrome in the background, wait, then it will save the PDF
    print_info "Waiting for MathJax to load and render (this may take a few seconds)..."
    
    # Use the newer headless mode which handles async rendering better
    (sleep 3 && "$chrome_bin" --headless=new --disable-gpu \
        --print-to-pdf="$output_pdf" \
        --print-to-pdf-options='{"headerTemplate": " ", "footerTemplate": "<div style=\"font-size: 8px; width: 100%; padding: 0 1cm; display: flex; justify-content: space-between;\"><span class=date></span><span><span class=pageNumber></span> / <span class=totalPages></span></span></div>", "displayHeaderFooter": true, "marginTop": "1.5cm", "marginBottom": "1.5cm"}' \
        --allow-file-access-from-files \
        --disable-web-security \
        "file://$html_file" 2>/dev/null) &
    
    wait $!
}

convert_with_latex() {
    local input_md="$1"
    local output_pdf="$2"
    
    print_info "Converting with LaTeX..."
    
    local engine=$(check_command xelatex && echo "xelatex" || echo "pdflatex")
    
    # Disable implicit_figures to prevent alt text from becoming captions
    pandoc "$input_md" \
        -f markdown+tex_math_dollars-implicit_figures \
        --pdf-engine="$engine" \
        -V geometry:margin=0.75in \
        -V fontsize=10pt \
        -V linestretch=1.3 \
        -V colorlinks=true \
        -V linkcolor=NavyBlue \
        --syntax-highlighting=pygments \
        ${VERBOSE:+--verbose} \
        -o "$output_pdf"
}

#===============================================================================
# Single Engine Conversion
#===============================================================================

convert_single() {
    local input_md="$1"
    local output_pdf="$2"
    
    TEMP_DIR=$(mktemp -d)
    
    local css_file="$TEMP_DIR/style.css"
    create_professional_css "$css_file"
    
    local processed_md="$TEMP_DIR/processed.md"
    process_images "$input_md" "$processed_md"
    
    case "$PDF_ENGINE" in
        weasyprint)
            convert_with_weasyprint "$processed_md" "$output_pdf" "$css_file"
            ;;
        chrome|chrome-macos|chromium|chromium-browser)
            convert_with_chrome "$processed_md" "$output_pdf" "$css_file"
            ;;
        xelatex|pdflatex)
            convert_with_latex "$processed_md" "$output_pdf"
            ;;
    esac
    
    if [[ -f "$output_pdf" ]]; then
        local size=$(du -h "$output_pdf" | cut -f1)
        print_success "PDF created successfully!"
        echo ""
        echo -e "${GREEN}📄 Output:${NC} $output_pdf"
        echo -e "${BLUE}📦 Size:${NC} $size"
        echo -e "${BLUE}🔧 Engine:${NC} $PDF_ENGINE"
        return 0
    else
        print_error "PDF generation failed"
        return 1
    fi
}

#===============================================================================
# Multi-Engine Conversion
#===============================================================================

convert_all_engines() {
    local input_md="$1"
    local base_output="$2"
    local base_name="${base_output%.pdf}"
    
    TEMP_DIR=$(mktemp -d)
    
    local css_file="$TEMP_DIR/style.css"
    create_professional_css "$css_file"
    
    local processed_md="$TEMP_DIR/processed.md"
    process_images "$input_md" "$processed_md"
    
    echo ""
    print_info "Generating PDFs with all available engines..."
    echo ""
    
    local generated_files=()
    local failed_engines=()
    
    # WeasyPrint
    if check_command weasyprint; then
        local output="${base_name}_weasyprint.pdf"
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_info "Engine: WeasyPrint"
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if convert_with_weasyprint "$processed_md" "$output" "$css_file" 2>&1; then
            if [[ -f "$output" ]]; then
                generated_files+=("$output:WeasyPrint")
                print_success "WeasyPrint conversion successful"
            else
                failed_engines+=("WeasyPrint:PDF file not created")
                print_error "WeasyPrint conversion failed - no output file"
            fi
        else
            failed_engines+=("WeasyPrint:Conversion error")
            print_error "WeasyPrint conversion failed"
        fi
        echo ""
    else
        print_warning "WeasyPrint not installed - skipping"
        echo ""
    fi
    
    # Chrome
    if check_command google-chrome || check_command chromium || check_command chromium-browser || check_chrome_macos; then
        local output="${base_name}_chrome.pdf"
        if check_command google-chrome; then
            PDF_ENGINE="chrome"
        elif check_chrome_macos; then
            PDF_ENGINE="chrome-macos"
        elif check_command chromium; then
            PDF_ENGINE="chromium"
        else
            PDF_ENGINE="chromium-browser"
        fi
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_info "Engine: Chrome/Chromium"
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if convert_with_chrome "$processed_md" "$output" "$css_file" 2>&1; then
            if [[ -f "$output" ]]; then
                generated_files+=("$output:Chrome")
                print_success "Chrome conversion successful"
            else
                failed_engines+=("Chrome:PDF file not created")
                print_error "Chrome conversion failed - no output file"
            fi
        else
            failed_engines+=("Chrome:Conversion error")
            print_error "Chrome conversion failed"
        fi
        echo ""
    else
        print_warning "Chrome/Chromium not installed - skipping"
        echo ""
    fi
    
    # LaTeX
    if check_command xelatex || check_command pdflatex; then
        local output="${base_name}_latex.pdf"
        PDF_ENGINE=$(check_command xelatex && echo "xelatex" || echo "pdflatex")
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_info "Engine: LaTeX ($PDF_ENGINE)"
        print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if convert_with_latex "$processed_md" "$output" 2>&1; then
            if [[ -f "$output" ]]; then
                generated_files+=("$output:LaTeX")
                print_success "LaTeX conversion successful"
            else
                failed_engines+=("LaTeX:PDF file not created")
                print_error "LaTeX conversion failed - no output file"
            fi
        else
            failed_engines+=("LaTeX:Conversion error")
            print_error "LaTeX conversion failed"
        fi
        echo ""
    else
        print_warning "LaTeX (xelatex/pdflatex) not installed - skipping"
        echo ""
    fi
    
    # Summary
    echo ""
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_success "Conversion Summary"
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [[ ${#generated_files[@]} -gt 0 ]]; then
        print_success "Successfully generated ${#generated_files[@]} PDF(s):"
        echo ""
        for file_info in "${generated_files[@]}"; do
            local file="${file_info%%:*}"
            local engine="${file_info##*:}"
            local size=$(du -h "$file" | cut -f1)
            echo -e "  ${GREEN}✓${NC} ${BLUE}$engine${NC}"
            echo -e "    📄 $file"
            echo -e "    📦 $size"
            echo ""
        done
    fi
    
    if [[ ${#failed_engines[@]} -gt 0 ]]; then
        echo ""
        print_warning "Failed conversions (${#failed_engines[@]}):"
        echo ""
        for fail_info in "${failed_engines[@]}"; do
            local engine="${fail_info%%:*}"
            local reason="${fail_info##*:}"
            echo -e "  ${RED}✗${NC} ${BLUE}$engine${NC}: $reason"
        done
        echo ""
    fi
    
    if [[ ${#generated_files[@]} -eq 0 ]]; then
        print_error "No PDFs were generated successfully"
        return 1
    fi
    
    return 0
}

#===============================================================================
# Main
#===============================================================================

main() {
    local input_md=""
    local output_pdf=""
    local VERBOSE=""
    local KEEP_TEMP=0
    local ALL_ENGINES=0
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help) show_usage; exit 0 ;;
            -v|--verbose) VERBOSE=1; shift ;;
            -k|--keep-temp) KEEP_TEMP=1; shift ;;
            -a|--all) ALL_ENGINES=1; shift ;;
            -*) print_error "Unknown option: $1"; show_usage; exit 1 ;;
            *)
                if [[ -z "$input_md" ]]; then
                    input_md="$1"
                elif [[ -z "$output_pdf" ]]; then
                    output_pdf="$1"
                else
                    print_error "Too many arguments"; exit 1
                fi
                shift
                ;;
        esac
    done
    
    [[ -z "$input_md" ]] && { print_error "No input file"; show_usage; exit 1; }
    [[ ! -f "$input_md" ]] && { print_error "File not found: $input_md"; exit 1; }
    
    [[ -z "$output_pdf" ]] && output_pdf="${input_md%.md}.pdf"
    [[ ! "$output_pdf" =~ \.pdf$ ]] && output_pdf="${output_pdf}.pdf"
    
    print_info "Markdown to PDF Converter v2.1.1"
    echo ""
    
    # Analyze document and recommend engine
    if [[ -n "${VERBOSE:-}" ]]; then
        recommend_engine "$input_md"
        echo ""
    fi
    
    if [[ $ALL_ENGINES -eq 1 ]]; then
        print_info "Mode: Compare all engines"
        convert_all_engines "$input_md" "$output_pdf" && exit 0 || exit 1
    else
        check_dependencies || exit 1
        print_info "Using: $PDF_ENGINE"
        echo ""
        convert_single "$input_md" "$output_pdf" && exit 0 || exit 1
    fi
}

main "$@"


