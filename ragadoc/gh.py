from github import Github
from typing import List
from urllib.parse import urlparse
from ragadoc.docs import DocInfo

class GitHubClient:

    def __init__(self, github_token: str):
        self.github = Github(github_token)

    def _parse_github_url(self, url: str) -> tuple[str, str, str, str]:
        url = url.rstrip("/")

        # parse URL parts
        parts = url.split("/tree/") if "/tree/" in url else [url, ""]
        repo_url = parts[0]
        path_parts = parts[1].split("/") if parts[1] else []

        # get owner and repo from main URL
        path = urlparse(repo_url).path.strip("/")
        owner, repo = path.split("/")[-2:]

        # extract branch and base_dir
        branch = path_parts[0] if path_parts else "main"
        base_dir = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""

        return owner, repo, branch, base_dir

    def fetch_docs(self, repo_urls: List[str]) -> List[DocInfo]:
        docs = []
        for url in repo_urls:
            print("url", url)
            owner, repo_name, branch, base_dir = self._parse_github_url(url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")

            try:
                contents = repo.get_contents(base_dir, ref=branch)
                if not isinstance(contents, list):
                    contents = [contents]

                while contents:
                    file_content = contents.pop(0)
                    if file_content.type == "dir":
                        path_contents = repo.get_contents(
                            file_content.path, ref=branch)
                        if not isinstance(path_contents, list):
                            path_contents = [path_contents]
                        contents.extend(path_contents)
                        print(f"extend contents ({len(contents)})")
                    elif file_content.path.endswith(('.md', '.mdx')):
                        docs.append(DocInfo(
                            path=file_content.path,
                            content=file_content.decoded_content.decode(),
                            sha=file_content.sha,
                            repo_name=f"{owner}/{repo_name}"
                        ))
                        print(
                            f"append to docs: {file_content.path} ({len(docs)})")
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                raise

        return docs
