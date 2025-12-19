# Learning Materials (`_blog` Collection)

This folder contains educational resources, tutorials, and learning materials created by the lab.

## Purpose

The `_blog` collection is separate from `_posts` (news/announcements) and is specifically for:
- **Tutorials**: Step-by-step guides on technical topics
- **Educational content**: Deep dives into research methods
- **Code examples**: Practical implementations
- **Learning resources**: Materials for students and researchers

## File Structure

```
_blog/
├── YYYY-MM-DD-tutorial-name.md
├── YYYY-MM-DD-another-tutorial.md
└── ...
```

## File Naming Convention

Files must follow this format:
```
YYYY-MM-DD-title-with-dashes.md
```

Examples:
- `2025-01-10-intro-federated-learning.md`
- `2025-01-05-lstm-energy-forecasting.md`
- `2025-01-15-physics-informed-neural-networks.md`

## File Template

```markdown
---
title: "Your Tutorial Title"
author: author-slug  # Must match a file in _members/
date: 2025-01-10
tags:
  - tag1
  - tag2
  - tag3
---

Brief introduction paragraph.

<!-- excerpt start -->
This text will appear in the listing page as a preview.
<!-- excerpt end -->

## Section 1

Your content here...

## Section 2

More content...

```

## Front Matter Fields

### Required Fields
- `title`: The tutorial/article title
- `author`: Author slug (matches filename in `_members/` folder, e.g., `gang-li`, `william-mayfield`)
- `date`: Publication date (YYYY-MM-DD format)

### Optional Fields
- `tags`: Array of tags for categorization
- `image`: Header image path (optional)

## Tags

Use descriptive tags to help users find content:

**Topic Tags:**
- `machine learning`, `deep learning`, `federated learning`, `LSTM`, `CNN`
- `renewable energy`, `wind energy`, `tidal energy`, `hydrogen`
- `predictive maintenance`, `condition monitoring`

**Content Type Tags:**
- `tutorial`, `guide`, `example`, `theory`
- `beginner`, `intermediate`, `advanced`
- `python`, `pytorch`, `tensorflow`

## Code Examples

Use proper syntax highlighting:

```python
def example_function():
    """Your code here"""
    return result
```

## Math Equations

Use LaTeX for equations:

Inline: `$E = mc^2$`

Block:
```
$$
f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx
$$
```

## Images

Place images in the `images/` folder and reference them:

```markdown
![Description](images/your-image.png)
```

## Creating New Learning Materials

1. **Choose a topic** relevant to lab research
2. **Create the file** in `_blog/` with proper naming
3. **Write comprehensive content** with examples
4. **Add code snippets** where applicable
5. **Include references** to papers and resources
6. **Test locally** before pushing

## Viewing Learning Materials

The learning materials appear on the **Learning Materials** page at `/learning/`

Users can:
- Browse all materials
- Search by keywords
- Filter by tags
- Read full tutorials

## Best Practices

### Content Quality
- ✅ Start with clear learning objectives
- ✅ Include practical examples
- ✅ Provide working code
- ✅ Link to relevant papers
- ✅ Add further reading sections

### Code Quality
- ✅ Test all code examples
- ✅ Use consistent formatting
- ✅ Add comments
- ✅ Include error handling
- ✅ Show complete examples

### Writing Style
- ✅ Be clear and concise
- ✅ Define technical terms
- ✅ Use headings for structure
- ✅ Include visuals when helpful
- ✅ Provide context and motivation

## Examples

See the existing files in this folder:
- `2025-01-10-intro-federated-learning.md` - Comprehensive FL tutorial
- `2025-01-05-lstm-energy-forecasting.md` - LSTM for time series

## Questions?

Contact the lab PI or check the main website documentation at `/README.md`
