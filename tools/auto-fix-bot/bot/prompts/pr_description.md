Generate a pull request description for the following bug fix.

## Issue
- Number: #{{ issue_number }}
- Title: {{ issue_title }}
- Component: {{ component }}

## Changes Made
{{ changes_summary }}

## Files Modified
{% for file in files_changed %}
- `{{ file.path }}` ({{ file.additions }} additions, {{ file.deletions }} deletions)
{% endfor %}

## Instructions
Write a PR description with these exact sections:
1. **Summary** — One paragraph explaining the fix
2. **Problem** — What was wrong and why
3. **Solution** — What was changed and why this approach was chosen
4. **Testing** — What tests were added, before/after behavior
5. **Note** (optional) — Any follow-up work or limitations

Format the output as markdown. Reference the issue number. Be concise but complete.
