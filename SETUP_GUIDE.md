# Gang Li Research Lab Website - Setup Guide

## Overview

This document provides a comprehensive guide to setting up and deploying the Gang Li Research Lab website based on the Lab Website Template.

## Website Structure

### Core Configuration Files

1. **_config.yaml** - Main Jekyll configuration
   - Site title: "Gang Li Research Lab"
   - Subtitle: "Intelligent Energy Systems & Advanced Mechanical Design"
   - Social links: Email, ORCID, Google Scholar, GitHub, Twitter, YouTube

### Content Pages

1. **index.md** - Homepage
   - Lab overview and mission
   - Three featured sections: Research, Projects, Team
   - Professional description of research focus

2. **research/index.md** - Research page
   - Four main research areas with detailed descriptions
   - Automatic publication import from ORCID
   - Featured publications section
   - Funding information

3. **projects/index.md** - Projects page
   - Featured and active projects
   - Collaboration partnerships
   - Technology transfer initiatives
   - Student opportunities

4. **team/index.md** - Team page
   - Principal investigator section
   - Ph.D. students section
   - Research focus areas
   - Recruitment information

5. **contact/index.md** - Contact page
   - Contact information
   - Location details
   - Information for prospective students and collaborators

6. **blog/index.md** - News & Updates page
   - Lab announcements and news
   - Searchable blog posts

### Team Member Profiles (_members/)

1. **gang-li.md** - Principal Investigator
   - Education background
   - Research interests
   - Recent honors and awards
   - Key funded projects

2. **william-mayfield.md** - Ph.D. Student
3. **zaki-moutassem.md** - Ph.D. Candidate
4. **sara-mouafik.md** - Ph.D. Student
5. **doha-bounaim.md** - Ph.D. Student

### Data Files (_data/)

1. **orcid.yaml** - ORCID ID for automatic publication import
2. **projects.yaml** - Project descriptions and metadata
   - 8 projects defined (3 featured, 5 active)
   - Tags for categorization

### Blog Posts (_posts/)

1. **2025-12-18-website-launch.md** - Website launch announcement
2. **2024-04-15-research-award.md** - Award announcement

## Key Features Implemented

### Research Integration
- Automatic citation import from ORCID (0000-0003-2793-4615)
- Google Scholar integration
- Publication search and filtering

### Project Showcase
- Featured projects with images and descriptions
- Tag-based categorization
- Collaboration highlights

### Team Management
- Individual profile pages for each member
- Role-based organization
- Contact information and links

### Content Organization
- Clear navigation structure
- Responsive design
- Search functionality

## Research Areas Covered

1. **AI-Enabled Predictive Maintenance**
   - Federated learning for distributed systems
   - Physics-informed neural networks
   - CNN-LSTM architectures

2. **Advanced Drivetrain Technologies**
   - Infinitely variable transmissions
   - Noncircular gear pairs
   - Control systems

3. **Energy Forecasting & Optimization**
   - Wind-to-hydrogen forecasting
   - LSTM networks
   - Physics-informed ensemble learning

4. **Gear Dynamics & Mechanical Systems**
   - Mesh stiffness modeling
   - Hypoid gear analysis
   - Digital twin technologies

## Deployment Instructions

### Step 1: Repository Setup
```bash
# Create new repository on GitHub named 'lab-website'
# Clone to local machine
git clone https://github.com/GangGavinLi/lab-website.git
cd lab-website
```

### Step 2: Copy Files
Copy all the generated files into your repository:
- _config.yaml
- index.md
- README.md
- All directories: _data/, _members/, _posts/, research/, projects/, team/, contact/, blog/

### Step 3: Install Dependencies
```bash
# Install Ruby and Bundler (if not already installed)
# Then install Jekyll dependencies
bundle install
```

### Step 4: Test Locally
```bash
# Run local server
bundle exec jekyll serve

# Visit http://localhost:4000 in browser
```

### Step 5: GitHub Setup
1. Push files to GitHub
2. Enable GitHub Pages in repository settings
3. Set source to "gh-pages" branch
4. Enable GitHub Actions

### Step 6: Enable Citation Import
The site will automatically import publications from ORCID on schedule.
To trigger manually:
- Go to Actions tab in GitHub
- Run "on-schedule" workflow

## Customization Guide

### Adding New Team Members
1. Create new file in `_members/` directory
2. Use existing files as templates
3. Add appropriate role and contact information

### Adding Projects
1. Edit `_data/projects.yaml`
2. Add new project with title, description, tags
3. Set `group: featured` for featured projects

### Adding Blog Posts
1. Create new file in `_posts/` directory
2. Use format: `YYYY-MM-DD-title.md`
3. Include frontmatter with title, author, tags

### Updating Research
Publications automatically update from ORCID.
For manual entries:
1. Edit `_data/sources.yaml`
2. Add DOI or citation information

### Styling Changes
1. Edit `_styles/-theme.scss` for colors and fonts
2. Modify section backgrounds in individual page files

## Important Information

### Current Content Status
- **Team**: 1 PI + 4 Ph.D. students defined
- **Projects**: 8 projects (3 featured)
- **Blog Posts**: 2 posts
- **Publications**: Imported from ORCID (60+ papers)

### Required Actions Before Going Live

1. **Add Images**
   - Replace placeholder `images/photo.jpg` with actual photos
   - Add lab photos, equipment, research images
   - Add team member headshots

2. **Update Links**
   - Verify all external links work
   - Add actual project URLs when available
   - Update social media handles if different

3. **Review Content**
   - Proofread all text
   - Verify accuracy of information
   - Check formatting on all pages

4. **Add More Team Members**
   - Include M.S. students
   - Add undergraduate researchers
   - Include alumni if desired

5. **Expand Blog**
   - Add more news posts
   - Include conference presentations
   - Share research highlights

### Maintenance Tasks

**Weekly**
- Check for new publications (automatic from ORCID)
- Update news with recent events

**Monthly**
- Review and update project statuses
- Add new team members as they join
- Check all links still work

**Semester**
- Update course information
- Refresh team member profiles
- Archive old news posts

## Technical Notes

### Jekyll Version
- Template requires Jekyll 4.3+
- Ruby 3.1+ recommended

### Citation System
- Uses Manubot for citation generation
- Automatic import from ORCID
- Citations cached for 90 days

### Search Functionality
- Client-side search using JavaScript
- No external services required
- Works with tags and content

### Analytics
- Placeholder for Google Analytics
- Edit `_includes/analytics.html` to add tracking code

## Support & Resources

- **Template Documentation**: https://greene-lab.gitbook.io/lab-website-template-docs
- **Jekyll Documentation**: https://jekyllrb.com/docs/
- **Markdown Guide**: https://www.markdownguide.org/

## Contact

For questions about this website:
- **Dr. Gang Li**: gli@me.msstate.edu
- **Lab Website**: (will be) https://ganggavinli.github.io/lab-website

---

**Document Version**: 1.0  
**Created**: December 18, 2025  
**Last Updated**: December 18, 2025
