def define_env(env):
    """
    Hook function for defining macros, variables and filters
    """

    @env.macro
    def render_gallery(gallery_items_param):
        """
        Renders a gallery of items with tag filtering

        Args:
            gallery_items_param: Either the gallery items list or string name of the variable

        Returns:
            HTML string for the gallery
        """
        # Get the actual gallery items list
        if isinstance(gallery_items_param, str):
            # If a string was passed, look it up in the environment variables
            if gallery_items_param in env.variables:
                gallery_items = env.variables[gallery_items_param]
            else:
                return f"<div class='error'>Error: Could not find variable '{gallery_items_param}'</div>"
        else:
            # Otherwise use the parameter directly
            gallery_items = gallery_items_param

        # Validate gallery_items is a list
        if not isinstance(gallery_items, list):
            return f"<div class='error'>Error: gallery_items is not a list, but a {type(gallery_items)}</div>"

        # Extract all unique tags from gallery items
        all_tags = []
        for item in gallery_items:
            if not isinstance(item, dict):
                continue

            if "tags" in item and item["tags"]:
                all_tags.extend(item["tags"])
        all_tags = sorted(list(set(all_tags)))

        # Generate HTML directly
        html = '<div class="examples-gallery-container">'

        # Generate tag filter select
        html += '<select multiple class="tag-filter" data-placeholder="Filter by tags">'
        for tag in all_tags:
            html += f'<option value="{tag}">{tag}</option>'
        html += "</select>"

        # Generate gallery cards
        html += '<div class="gallery-cards">'

        for item in gallery_items:
            # Skip if item is not a dictionary
            if not isinstance(item, dict):
                continue

            image_url = item.get("image", "default.png")
            if image_url and not isinstance(image_url, str):
                # Handle case where image is not a string
                image_url = "default.png"

            if image_url and not image_url.startswith("http"):
                image_url = f"../../../../assets/img/gallery/{image_url}"

            # Handle default image
            if not image_url:
                image_url = "../../../../assets/img/gallery/default.png"

            # Tags HTML
            tags_html = ""
            if "tags" in item and item["tags"]:
                tags_html = '<div class="tags-container">'
                for tag in item["tags"]:
                    tags_html += f'<span class="tag" data-tag="{tag}">{tag}</span>'
                tags_html += "</div>"

            # Badges HTML
            badges_html = ""
            if "source" in item and item["source"]:
                colab_href = f"https://colab.research.google.com/github/ag2ai/ag2/blob/main/{item['source']}"
                github_href = f"https://github.com/ag2ai/ag2/blob/main/{item['source']}"
                badges_html = f"""
                <span class="badges">
                    <a style="margin-right: 5px" href="{colab_href}" target="_parent">
                        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
                    </a>
                    <a href="{github_href}" target="_parent">
                        <img alt="GitHub" src="https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github"/>
                    </a>
                </span>
                """

            # Generate card HTML with safer access to attributes
            tags_str = ",".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else ""

            html += f"""
            <div class="card" data-link="{item.get("link", "#")}" data-tags="{tags_str}">
                <div class="card-container">
                    <img src="{image_url}" alt="{item.get("title", "")}" class="card-image">
                    <p class="card-title">{item.get("title", "")}</p>
                    {badges_html}
                    <p class="card-description">{item.get("description", item.get("title", ""))}</p>
                    {tags_html}
                </div>
            </div>
            """

        # Close containers
        html += """
            </div>
        </div>
        """

        return html
