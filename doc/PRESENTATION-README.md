# MLIR AOT Compilation Demo - Presentation Guide

This directory contains presentation materials for the MLIR AOT compilation pipeline demo.

## Files

- **DEMO.md** - Detailed presentation guide with timing markers for 20-30 min tech meeting
- **DEMO-SLIDES.md** - Marp-based slide deck (markdown format)

## Creating Presentation Slides

The slides are written in [Marp](https://marp.app/) markdown format and can be converted to:
- PDF slides
- PowerPoint (PPTX)
- HTML slides
- PNG images

### Option 1: Using Marp CLI (Recommended)

#### Install Marp CLI

```bash
# Using npm
npm install -g @marp-team/marp-cli

# Or use npx (no installation needed)
npx @marp-team/marp-cli --version
```

#### Convert to Different Formats

```bash
# Convert to PDF
marp doc/DEMO-SLIDES.md --pdf

# Convert to PowerPoint
marp doc/DEMO-SLIDES.md --pptx

# Convert to HTML
marp doc/DEMO-SLIDES.md --html

# Convert to PNG images (one per slide)
marp doc/DEMO-SLIDES.md --images png

# Preview in browser (live reload)
marp doc/DEMO-SLIDES.md --preview
```

#### Specify Output Location

```bash
# Save to specific file
marp doc/DEMO-SLIDES.md --pdf -o presentations/MLIR-AOT-Demo.pdf

# Save to specific directory
marp doc/DEMO-SLIDES.md --pptx -o presentations/
```

### Option 2: Using Marp for VS Code

1. Install the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension
2. Open `doc/DEMO-SLIDES.md` in VS Code
3. Click the preview icon in the top-right corner
4. Export using the command palette: `Marp: Export slide deck...`

### Option 3: Using Docker (No Installation)

```bash
# Run Marp in Docker container
docker run --rm -v $PWD:/home/marp/app/ marpteam/marp-cli \
  doc/DEMO-SLIDES.md --pdf -o presentations/MLIR-AOT-Demo.pdf
```

## Customizing the Slides

### Change Theme

Edit the frontmatter in `DEMO-SLIDES.md`:

```yaml
---
marp: true
theme: default  # Options: default, gaia, uncover
---
```

### Change Colors

Modify the `style` section in the frontmatter:

```yaml
style: |
  section {
    font-size: 28px;
  }
  h1 {
    color: #0066cc;  # Change title color
  }
```

### Add Images

```markdown
![width:600px](path/to/image.png)
```

### Two-Column Layout

```markdown
<div class="columns">
<div>

**Left column content**

</div>
<div>

**Right column content**

</div>
</div>
```

## Presentation Tips

### For 20-30 Minute Tech Meeting

1. **Opening Hook (Slides 1-3, ~2 min)** - Set context quickly
2. **Live Demo (Slides 4-6, ~5 min)** - Show the commands working
3. **Pipeline Breakdown (Slides 7-21, ~10 min)** - Walk through transformations
4. **Key Innovations (Slides 22-26, ~5 min)** - Highlight unique aspects
5. **Status & Next Steps (Slides 27-30, ~3 min)** - Progress overview
6. **Try It Yourself (Slides 31-35, ~2 min)** - Hands-on commands
7. **Q&A (Slides 36-38)** - Questions and wrap-up

### Presentation Mode Tips

- Use **speaker notes** for detailed talking points (see DEMO.md)
- Have the **live demo** terminal ready before presenting
- Pre-run commands to ensure they work on your system
- Keep **output files** ready to show (`../output/demo_stage*.mlir`)

### Live Demo Setup

Before presenting:

```bash
# Build the compiler
cd /path/to/onnx-hipdnn-ep
cmake -S . -B ../../build/onnx-hipdnn-ep -DBUILD_HIP_OPT_TOOL=ON
cmake --build ../../build/onnx-hipdnn-ep --config Debug --target hip-opt

# Test all three stages
../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip

../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip --convert-hip-to-llvm

../../build/onnx-hipdnn-ep/bin/hip-opt.exe \
  tools/hip-opt/demo_two_layer_conv.mlir \
  --convert-onnx-to-hip --generate-interface
```

## Updating the Presentation

When updating the demo:

1. **Update DEMO.md first** - This is the master source with full details
2. **Update DEMO-SLIDES.md** - Keep slides concise (code excerpts only)
3. **Regenerate exports** - Run `marp` to create new PDF/PPTX
4. **Test commands** - Ensure all demo commands still work
5. **Update status section** - Reflect current implementation progress

## Resources

- [Marp Documentation](https://marpit.marp.app/)
- [Marp CLI Documentation](https://github.com/marp-team/marp-cli)
- [Marp Themes](https://github.com/marp-team/marp-core/tree/main/themes)
- [Markdown Syntax](https://www.markdownguide.org/basic-syntax/)
