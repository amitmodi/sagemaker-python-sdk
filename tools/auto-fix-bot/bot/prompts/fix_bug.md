## Task: Fix Bug #{{ issue_number }}

**Title**: {{ issue_title }}

**Component**: {{ component }}

**Labels**: {{ labels }}

**Bug Report**:
```
{{ issue_body }}
```

## Instructions

1. **Understand the bug**: Read the issue carefully. Identify:
   - What is the expected behavior?
   - What is the actual behavior?
   - What error/exception occurs?
   - Is there a reproduction script?

2. **Find the relevant code**: The issue mentions specific classes/modules. Find them:
   - Search for the class name in the source code
   - Identify the file(s) that need to change
   - Look at the test directory for that module

3. **Find existing patterns**: Before making changes, search for how similar things are done:
   - If it's a type annotation issue, find how other fields handle the same type
   - If it's a missing feature, find where the feature exists in other classes
   - Use `search_files` to find patterns like the fix you need

4. **Make the fix**:
   - Change only the minimum needed
   - Follow the existing code style exactly
   - Add necessary imports

5. **Add tests**:
   - Find the test directory for the affected module
   - Look at existing test patterns
   - Add a test that would have caught this bug
   - Test both the fix and edge cases

6. **Validate**:
   - Run the specific test file with pytest
   - Ensure no existing tests are broken

## Important

- The repo root is at: `{{ repo_path }}`
- Only modify files under the repo root
- Prefer modifying existing test files over creating new ones (unless the test is for a new area)
- Report your confidence level (0.0-1.0) when done
