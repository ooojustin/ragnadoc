from github import Github
from typing import List, Optional
from urllib.parse import urlparse
from ragnadoc.docs import DocInfo
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn


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

    def fetch_docs(self, repo_urls: List[str], show_progress: Optional[bool] = True) -> List[DocInfo]:
        docs = []

        progress = None
        fetch_task = None

        if show_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            progress.start()
            fetch_task = progress.add_task(
                "Fetching documents...", total=None)

        try:
            for url in repo_urls:
                owner, repo_name, branch, base_dir = self._parse_github_url(
                    url)

                if progress and fetch_task is not None:
                    progress.update(
                        fetch_task, description=f"Fetching: {owner}/{repo_name}")

                # if progress:
                #     progress.console.log(
                #         f"Processing repository: [blue]{url}[/blue]")

                owner, repo_name, branch, base_dir = self._parse_github_url(
                    url)
                repo_path = f"{owner}/{repo_name}"
                repo = self.github.get_repo(repo_path)

                try:
                    contents = repo.get_contents(base_dir, ref=branch)
                    assert isinstance(contents, list)

                    repo_task = None
                    if progress:
                        repo_task = progress.add_task(
                            f"Fetching files from {repo_name}", total=len(contents))

                    while contents:
                        file_content = contents.pop(0)
                        if progress and repo_task is not None:
                            progress.update(
                                repo_task, advance=1, description=f"Processing {file_content.path}")

                        if file_content.type == "dir":
                            dir_contents = repo.get_contents(
                                file_content.path, ref=branch)
                            if not isinstance(dir_contents, list):
                                dir_contents = [dir_contents]
                            contents.extend(dir_contents)
                            if progress and repo_task is not None:
                                progress.update(repo_task, total=len(
                                    contents) + progress.tasks[repo_task].completed)
                        elif file_content.path.endswith((".md", ".mdx")):
                            # TODO: support more file types
                            docs.append(DocInfo(
                                path=file_content.path,
                                content=file_content.decoded_content.decode(),
                                sha=file_content.sha,
                                repo_name=repo_path
                            ))
                            if progress:
                                progress.console.log(
                                    f"[green]Added:[/green] {repo_path}/{file_content.path}")

                    if progress and repo_task is not None:
                        progress.update(
                            repo_task, advance=1, description=f"Processed repository: [blue]{repo_path}[/blue]")

                    if progress and fetch_task is not None:
                        progress.update(fetch_task, advance=1)
                except Exception as e:
                    if progress:
                        progress.console.log(
                            f"[red]Error processing {url}: {e}[/red]")
                    raise

            # mark fetch task as complete at the end
            if progress and fetch_task is not None:
                progress.remove_task(fetch_task)
                # progress.update(
                #     fetch_task, description="All repositories processed")
                # fetch_task, description="All repositories processed", completed=len(repo_urls))

        finally:
            if progress:
                progress.console.log(
                    f"[blue]Done:[/blue] Repository indexing completed.")
                progress.stop()

        return docs
