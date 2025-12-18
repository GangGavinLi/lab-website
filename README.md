![on-push](../../actions/workflows/on-push.yaml/badge.svg)
![on-pull-request](../../actions/workflows/on-pull-request.yaml/badge.svg)
![on-schedule](../../actions/workflows/on-schedule.yaml/badge.svg)

# Gang Li Research Lab Website

Visit **[ganggavinli.github.io/lab-website-template](https://ganggavinli.github.io/lab-website-template)** 🚀

_Built with [Lab Website Template](https://greene-lab.gitbook.io/lab-website-template-docs)_

## About

This is the official website for the Gang Li Research Lab at Mississippi State University. Our lab focuses on:

- AI-enabled predictive maintenance for renewable energy systems
- Advanced drivetrain design for tidal and wind energy converters
- Federated learning for distributed energy monitoring
- Time-series forecasting for hydrogen production
- Gear dynamics and mechanical power transmission

## Features

- **Automatic Citation Import**: Publications are automatically imported from ORCID (0000-0003-2793-4615)
- **Responsive Design**: Works beautifully on all devices
- **Easy to Update**: Simple markdown files for content management
- **SEO Optimized**: Built-in search engine optimization
- **GitHub Actions**: Automatic deployment and updates

## Quick Start

1. **Clone this repository**
   ```bash
   git clone https://github.com/GangGavinLi/lab-website.git
   cd lab-website
   ```

2. **Install dependencies** (requires Ruby and Bundler)
   ```bash
   bundle install
   ```

3. **Run locally**
   ```bash
   bundle exec jekyll serve
   ```
   
4. Visit `http://localhost:4000` in your browser

## Updating Content

### Adding Team Members

Create a new file in `_members/` directory:

```yaml
---
name: Your Name
image: images/photo.jpg
role: phd  # or postdoc, undergrad, etc.
links:
  email: your.email@example.com
  github: yourusername
---

Your bio goes here...
```

### Adding Publications

Publications are automatically imported from ORCID. To add manual citations, edit `_data/sources.yaml`:

```yaml
- id: doi:10.1234/example
  type: paper
  description: Optional description
  tags:
    - renewable energy
    - AI
```

### Adding Projects

Edit `_data/projects.yaml`:

```yaml
- title: Project Name
  subtitle: Brief subtitle
  group: featured  # optional
  image: images/photo.jpg
  description: Project description
  tags:
    - tag1
    - tag2
```

### Adding Blog Posts

Create a new file in `_posts/` directory:

```markdown
---
title: Post Title
author: author-slug
tags:
  - renewable energy
  - research
---

Your post content...
```

## Directory Structure

```
├── _config.yaml          # Site configuration
├── _data/               # Data files (projects, citations, etc.)
├── _members/            # Team member profiles
├── _posts/              # Blog posts
├── contact/             # Contact page
├── images/              # Images and media
├── index.md             # Homepage
├── projects/            # Projects page
├── research/            # Research page
└── team/                # Team page
```

## Configuration

Main settings are in `_config.yaml`:

```yaml
title: Gang Li Research Lab
subtitle: Intelligent Energy Systems & Advanced Mechanical Design
description: Your lab description

links:
  email: gli@me.msstate.edu
  orcid: 0000-0003-2793-4615
  google-scholar: ETJoidYAAAAJ
  github: GangGavinLi
```

## Customization

### Colors and Theme

Edit `_styles/-theme.scss` to customize colors, fonts, and styling.

### Adding Pages

Create a new markdown file with frontmatter:

```markdown
---
title: Page Title
nav:
  order: 5
  tooltip: Page description
---

Your content...
```

### Images

Add images to the `images/` directory and reference them:

```markdown
![Alt text](images/your-image.jpg)
```

## Deployment

This site automatically deploys to GitHub Pages when you push to the main branch. Make sure:

1. GitHub Pages is enabled in repository settings
2. Source is set to "gh-pages" branch
3. GitHub Actions have appropriate permissions

## Support

For template documentation, visit: https://greene-lab.gitbook.io/lab-website-template-docs

For lab-specific questions, contact: gli@me.msstate.edu

## Citation

If you use this website template, please cite:

```
Lab Website Template
https://github.com/greenelab/lab-website-template
```

## License

This website is based on the Lab Website Template, which is licensed under BSD 3-Clause License.

---

**Gang Li Research Lab**  
Michael W. Hall School of Mechanical Engineering  
Mississippi State University  
Mississippi State, MS 39762
