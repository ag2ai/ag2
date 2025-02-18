## Define mini-templates for each portion of the doco.

<%!
  import re

  import pdoc

  from autogen.doc_utils import _PDOC_MODULE_EXPORT_MAPPINGS

  link_prefix = ''
  show_inherited_members = True

  def make_link_with_symbol_names(dobj: pdoc.Doc, current_module, name=None) -> str:
    if isinstance(dobj, pdoc.External):
        # Get last part of the name - the symbol itself
        full_name = dobj.name
        parts = full_name.split('.')
        return parts[-1]

    # For internal objects, extract the simple name too
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')

    # If it's a type from a qualified path, extract just the type name
    if '.' in name:
        parts = name.split('.')
        return parts[-1]

    return name

  def make_link(dobj: pdoc.Doc, current_module, name=None) -> str:
    """Mirror the HTML template's link function but generate markdown links.

    Args:
        dobj: The doc object to link to.
        name: The name to display for the link. If None, the doc object's qualname is used.

    Returns:
        [text](url) format instead of <a> tags.
    """
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')
    if isinstance(dobj, pdoc.External):
        fullname = f"{dobj.module}.{dobj.name}" if dobj.module else dobj.name
        if fullname in _PDOC_MODULE_EXPORT_MAPPINGS:
            symbol_name = fullname.split('.')[-1]
            new_path = _PDOC_MODULE_EXPORT_MAPPINGS[fullname]
            url = f"/docs/api-reference/{new_path.replace('.','/')}/{symbol_name}"
            return f'[{symbol_name}]({url})'
        else:
            return name

    elif 'autogen' in dobj.refname:
        if dobj.refname in _PDOC_MODULE_EXPORT_MAPPINGS:
            symbol_name = fullname.split('.')[-1]
            new_path = _PDOC_MODULE_EXPORT_MAPPINGS[fullname]
            url = f"/docs/api-reference/{new_path.replace('.','/')}/{symbol_name}"
            return f'[{symbol_name}]({url})'
        else:
            symbol_name = dobj.refname.split('.')[-1]
            url = f"/docs/api-reference/{dobj.refname.replace('.', '/')}"
            return f'[{symbol_name}]({url})'

    url = dobj.url(relative_to=current_module, link_prefix=link_prefix,
                    top_ancestor=not show_inherited_members)
    return f'[{name}]({url})'

  def indent(s, spaces=4):
      new = s.replace('\n', '\n' + ' ' * spaces)
      return ' ' * spaces + new.strip()

  def extract_param_descriptions(docstring):
      """Extract parameter descriptions from docstring."""
      param_desc = {}
      if not docstring or 'Args:' not in docstring:
          return param_desc

      # Extract the Args section
      parts = docstring.split('Args:', 1)
      if len(parts) < 2:
          return param_desc

      args_section = parts[1]

      # Find where the Args section ends
      for section in ['Returns:', 'Raises:', 'Note:', 'Example:', 'Examples:']:
          if f'\n{section}' in args_section:
              args_section = args_section.split(f'\n{section}')[0]

      # Split into lines and process
      lines = args_section.split('\n')
      current_param = None
      current_desc = []
      param_indent = None

      for line in lines:
          if not line.strip():
              continue

          # Calculate indentation
          indent_level = len(line) - len(line.lstrip())
          stripped = line.strip()

          # Check if this is a parameter definition line
          if ':' in stripped:
              # If this is a new parameter (either first one or less/equal indentation than previous)
              if param_indent is None or indent_level <= param_indent:
                  # Save previous parameter if exists
                  if current_param and current_desc:
                      param_desc[current_param] = ' '.join(current_desc).strip()

                  # Parse new parameter
                  param_parts = stripped.split(':', 1)
                  param_name = param_parts[0].split('(')[0].strip()
                  desc = param_parts[1].strip()

                  current_param = param_name
                  current_desc = [desc] if desc else []
                  param_indent = indent_level
                  continue

          # Add to current description if we have a parameter and indentation is valid
          if current_param and (param_indent is None or indent_level >= param_indent):
              current_desc.append(stripped)

      # Don't forget to save the last parameter
      if current_param and current_desc:
          param_desc[current_param] = ' '.join(current_desc).strip()

      return param_desc

  def format_param_table(params, obj, docstring):
      # remove self and * from params
      params = [param for param in params if not param.startswith('self:') and param != 'self' and param != '*']

      if not params:
          return ""

      param_descriptions = extract_param_descriptions(docstring)
      table = "| Name | Description |\n|--|--|\n"

      for param in params:
          # Split the parameter into name and type annotation
          parts = param.split(':')
          if len(parts) > 1:
              name = parts[0].strip()
              type_default = parts[1].strip()
              # Handle default values
              if '=' in type_default:
                  type_val, default = type_default.rsplit('=', 1)
                  type_val = type_val.strip().replace('|', '\\|')
                  default = default.strip().replace('|', '\\|')
              else:
                  type_val = type_default.replace('|', '\\|')
                  default = '-'

              type_val = type_val if type_val else '-'
          else:
              name = param.strip()
              type_val = '-'
              default = '-'

          # Get description from docstring and format it
          description = param_descriptions.get(name, "")
          # Convert multiple spaces to single space but preserve intended line breaks
          description = ' '.join(description.split())
          # Escape { and < characters to prevent it from being interpreted as special markdown characters
          description = description.replace('{', '\{').replace("<", "").replace("```python ", "```")
          default = default.replace('{', '\{').replace("<", "")
          # Add line breaks before numbered points
          description = re.sub(r'(?<!\d)\. ', '.<br/><br/>', description)


          # Format the table cell
          formatted_desc = f"{description}<br/><br/>" if description else ''
          if type_val != '-':
            ##   formatted_desc += f"**Type:** {get_doc_path(type_val)}"
              type_val = re.sub(r"<class[\s\xa0]'([^']+)'>", r"'\1'", type_val)
              formatted_desc += f"**Type:** {type_val}"
          if default != '-' and default != '"-"':
              formatted_desc += f"<br/><br/>**Default:** {default}"

          table += f"| `{name}` | {formatted_desc} |\n"
          ret_val = "<b>Parameters:</b>" + "\n" + table

      return ret_val
%>

<%!
def clean_docstring(text):
    """Format docstring text with consistent line breaks and safe characters.

    Args:
        text: The docstring text to clean

    Returns:
        Cleaned text with:
        - Escaped special characters
        - Single newlines after periods/colons converted to HTML breaks
        - Code blocks properly formatted
    """
    if not text:
        return ''

    # Step 1: Escape special characters that could cause rendering issues
    text = text.replace('{', '\\{')  # Escape curly braces

    # Enclose angle brackets like <agent> or <some.text> that aren't HTML tags (e.g., <item />) or already in ``
    text = re.sub(r'(?<!`)<([^/\s>]+)(?:\s[^>]*)?>(?!`)', lambda m: f'`{m.group(0)}`', text)

    # Step 2: Convert single newlines to HTML breaks, but preserve paragraphs
    # Period followed by newline -> period + break
    text = re.sub(r'\.\s*\n(?!\n)', '.<br/>', text)

    # Colon followed by newline -> colon + break
    text = re.sub(r':\s*\n(?!\n)', ': <br/>', text)

    # Step 3: Fix code block formatting
    # Ensure code blocks start on new lines, not after breaks
    text = text.replace('<br/>```python', '\n```python')

    return text
%>

<%def name="deflist(s)">
% if 'Args:' in s:
${indent(s.split('Args:')[0])}
% elif 'Attributes:' in s:
${indent(s.split('Attributes:')[0])}
% else:
${indent(s)}
% endif
</%def>

<%def name="h3(s)">### ${s}
</%def>

<%def name="h2(s)">## ${s}
</%def>

<%!
  # Add this new function with the existing helper functions
  def extract_return_description(docstring):
      """Extract return description from docstring."""
      if not docstring or 'Returns:' not in docstring:
          return ""

      # Split on Returns: to get the section after it
      post_returns = docstring.split('Returns:', 1)[1]

      # Look for the next section marker
      next_section_split = [post_returns.split(f"\n{marker}")[0]
                          for marker in ['Args:', 'Raises:', 'Note:', 'Example:', 'Examples:', 'Attributes:']
                          if f"\n{marker}" in post_returns]

      # If we found another section, use the text up to that section
      # Otherwise use all the text after Returns
      description = next_section_split[0] if next_section_split else post_returns

      return ' '.join(description.split()).strip()

  def format_returns_table(returns, docstring):
      if not returns:
          return ""

      description = extract_return_description(docstring)
      if not description:
          return ""

      table = "| Type | Description |\n|--|--|\n"
      returns = returns.replace('{', '\{').replace("<", "&lt;").replace("|", "\\|")
      description = description.replace('{', '\{').replace("<", "&lt;").replace("|", "\\|")

      table += f"| {returns} | {description} |\n"
      return "<b>Returns:</b>" + "\n" + table
%>

<%def name="function(func)" buffered="True">

<%
    metadata = "" if func.cls else f"""
---
sidebarTitle: {func.name}
title: {func.module.name}.{func.name}
---
"""
%>

${metadata}

<code class="doc-symbol doc-symbol-heading doc-symbol-${func.cls and 'method' or 'function'}"></code>
${'####'} ${func.name}

<%
        link = lambda dobj, name=None: make_link(dobj, func.module, name)
        params_with_symbol_links = func.params(annotate=show_type_annotations, link=link)
        returns_with_symbol_links = show_type_annotations and func.return_annotation(link=link) or ''

        link_with_symbol_name = lambda dobj, name=None: make_link_with_symbol_names(dobj, func.module, name)
        params_without_symbol_links = func.params(annotate=show_type_annotations, link=link_with_symbol_name)
        returns_without_symbol_links = show_type_annotations and func.return_annotation(link=link_with_symbol_name) or ''

        if len(params_without_symbol_links) > 2:
            formatted_params = ',\n    '.join(params_without_symbol_links)
            signature = f"{func.name}(\n    {formatted_params}\n) -> {returns_without_symbol_links}"
        else:
            signature = f"{func.name}({', '.join(params_without_symbol_links)}) -> {returns_without_symbol_links}"

        cleaned_docstring = clean_docstring(func.docstring)
%>
```python
${signature}
```

${cleaned_docstring | deflist}

% if len(params_with_symbol_links) > 0:
${format_param_table(params_with_symbol_links, func, func.docstring)}
% endif

% if returns_with_symbol_links:
${format_returns_table(returns_with_symbol_links, func.docstring)}
% endif

<br />
</%def>

<%def name="variable(var)" buffered="True">

<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
${'####'} ${var.name}
<br />
<%
        annot = show_type_annotations and var.type_annotation() or ''
        if annot:
            annot = f"({annot}) "

        cleaned_docstring = clean_docstring(var.docstring)
        if not cleaned_docstring:
            cleaned_docstring = '<br />'
%>
${cleaned_docstring | deflist}
</%def>

<%def name="class_(cls)" buffered="True">
---
sidebarTitle: ${cls.name}
title: ${cls.module.name}.${cls.name}
---
<h2 id="${cls.module.name}.${cls.name}" class="doc doc-heading">
    <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
    <span class="doc doc-object-name doc-class-name">${cls.name}</span>
</h2>

<%
    link = lambda dobj, name=None: make_link(dobj, cls.module, name)
    params_with_symbol_links = cls.params(annotate=show_type_annotations, link=link)

    link_with_symbol_name = lambda dobj, name=None: make_link_with_symbol_names(dobj, cls.module, name)
    params_without_symbol_links = cls.params(annotate=show_type_annotations, link=link_with_symbol_name)


    if len(params_without_symbol_links) > 2:
        formatted_params = ',\n    '.join(params_without_symbol_links)
        signature = f"{cls.name}(\n    {formatted_params}\n)"
    else:
        signature = f"{cls.name}({', '.join(params_without_symbol_links)})"

    cleaned_docstring = clean_docstring(cls.docstring)
%>

```python
${signature}
```
${cleaned_docstring | deflist}

% if len(params_with_symbol_links) > 0:
${format_param_table(params_with_symbol_links, cls, cleaned_docstring)}
% endif

<%
  class_vars = cls.class_variables(show_inherited_members, sort=sort_identifiers)
  static_methods = cls.functions(show_inherited_members, sort=sort_identifiers)
  inst_vars = cls.instance_variables(show_inherited_members, sort=sort_identifiers)
  methods = cls.methods(show_inherited_members, sort=sort_identifiers)
  mro = cls.mro()
  subclasses = cls.subclasses()
%>
% if class_vars:
${'###'} Class Attributes
    % for v in class_vars:
${variable(v)}

    % endfor
% endif
% if static_methods:
${'###'} Static Methods
    % for f in static_methods:
${function(f)}

    % endfor
% endif
% if inst_vars:
${'###'} Instance Attributes
    % for v in inst_vars:
${variable(v)}

    % endfor
% endif
% if methods:
${'###'} Instance Methods
    % for m in methods:
${function(m)}

    % endfor
% endif
</%def>

## Start the output logic for an entire module.

<%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  heading = 'Namespace' if module.is_namespace else 'Module'
  symbol_name = module.name.split('.')[-1]

  # filter out if module name is not the same
  classes = [c for c in classes if c.__module__ != "autogen"]
%>

% if submodules:
**** SUBMODULE_START ****
---
sidebarTitle: overview
title: ${module.name}
---

${h2('Sub-modules')}
    % for m in submodules:
* ${m.name}
    % endfor
**** SUBMODULE_END ****
% endif

% if variables:
${h2('Variables')}
    % for v in variables:
${variable(v)}

    % endfor
% endif

% if functions:
${h2('Functions')}
    % for f in functions:
**** SYMBOL_START ****
${function(f)}
**** SYMBOL_END ****

    % endfor
% endif

% if classes:
    % for c in classes:
**** SYMBOL_START ****
${class_(c)}
**** SYMBOL_END ****

    % endfor
% endif
