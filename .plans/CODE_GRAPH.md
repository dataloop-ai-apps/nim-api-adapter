# Code Structure Index

**35 files | 44 classes | 50 functions | 183 methods | 267 call edges | 0 import edges**

## agent/__init__.py (0 symbols)


## agent/downloadables_create.py (14 symbols)

fn _get_agent_dir() -> Path [L27-L29]
fn _get_repo_root() -> Path [L32-L34]
  calls: _get_agent_dir
fn _extract_version() -> str [L37-L44]
  calls: _get_repo_root
fn model_name_from_downloadable_dpk_name(dpk_name: str) -> str | None [L53-L73]
fn _model_name_from_manifest_path(manifest_path: str) -> str [L76-L87]
fn _alnum_suffix(manifest_path: str, length: int = _RANDOM_SUFFIX_LEN) -> str [L90-L93]
fn _component_names_from_manifest_path(manifest_path: str) -> dict [L96-L109]
  calls: _model_name_from_manifest_path, _alnum_suffix
fn _get_nim_entrypoint(model_name: str) -> str [L112-L139]
  calls: run
fn create_manifest(
    model_name: str,
    manifest_path: str,
    image_version: str = "0.1.13",
    runner_image_override: str | None = None
) -> dict [L142-L206]
  calls: _get_agent_dir, _component_names_from_manifest_path, _get_api_model_name, _get_repo_root
fn build_docker_image(model_name: str, image_version: str = "1.0.0") -> str [L209-L278]
  calls: _get_agent_dir, _get_nim_entrypoint, run
fn _get_api_model_name(manifest_path: str) -> str | None [L281-L291]
  calls: _get_repo_root, load
fn _get_existing_runner_image(manifest_path: str) -> str | None [L294-L314]
  calls: _get_repo_root, load
fn build_downloadable_nim(model_name: str, manifest_path: str, skip_docker: bool = False) -> dict [L317-L372]
  calls: _extract_version, _get_existing_runner_image, build_docker_image, create_manifest
fn fix_existing_downloadable_manifests() -> int [L375-L408]
  calls: _get_repo_root, load, _component_names_from_manifest_path, _get_api_model_name

## agent/dpk_mcp_handler.py (15 symbols)

fn get_dpk_version(use_github: bool = True) -> str [L37-L79]
  calls: GitHubClient, _get_file_content
fn parse_model_id(model_id: str) -> tuple[str, str] [L97-L113]
fn model_to_dpk_name(model_id: str) -> str [L116-L120]
fn get_model_provider(model_id: str) -> str [L123-L182]
fn get_adapter_path(model_type: str) -> str [L185-L188]
fn get_model_folder(model_id: str, model_type: str) -> str [L191-L199]
  calls: parse_model_id
fn get_manifest_path(model_id: str, model_type: str) -> str [L202-L204]
  calls: get_model_folder
class DPKGeneratorClient [L246-L469]
  """MCP Client for generating DPK manifests for NVIDIA NIM models.

    Supports two..."""
  .__init__(self) [L257-L266]
    calls: _resolve_dpk_url
  @staticmethod
  ._resolve_dpk_url(dpk_name: str) -> str [L269-L280]
  ._http_call_tool(self, tool_name: str, arguments: dict) -> dict [L282-L315]
  ._stdio_call_tool(self, tool_name: str, arguments: dict) -> dict [L317-L330]
  ._call_tool(self, tool_name: str, arguments: dict) -> dict [L332-L336]
    calls: _http_call_tool, run, _stdio_call_tool
  .create_nim_dpk_manifest(self, model_id: str, model_type: str, embeddings_size: int = None, license: str = None) -> dict [L338-L469]
    calls: model_to_dpk_name, get_model_provider, get_dpk_version, _call_tool

fn create_nim_manifest(model_id: str, model_type: str, embeddings_size: int = None) -> dict [L476-L479]
  calls: DPKGeneratorClient, create_nim_dpk_manifest

## agent/github_client.py (15 symbols)

class GitHubClient [L44-L593]
  """Client for GitHub operations - creating branches, commits, and PRs.
    
    Cre..."""
  .__init__(
        self,
        token: str = None,
        repo: str = None,
        base_branch: str = "main"
    ) [L51-L82]
  @property
  .client(self) [L85-L89]
  @property
  .repository(self) [L92-L96]
  ._find_model_folder_by_dpk_name(self, dpk_name: str, model_type: str) -> Optional[str] [L102-L152]
    calls: _find_all_manifests_in_path, _get_file_content
  ._find_all_manifests_in_path(self, base_path: str) -> list [L154-L176]
    calls: _find_all_manifests_in_path
  ._get_file_content(self, path: str, branch: str = None) -> Optional[str] [L182-L191]
  ._update_dataloop_cfg(self, existing_content: str, new_manifest_paths: List[str], deprecated_manifest_paths: List[str] = None) -> str [L193-L223]
  ._update_bumpversion_cfg(self, existing_content: str, new_manifest_paths: List[str], deprecated_manifest_paths: List[str] = None) -> str [L225-L277]
  .create_new_and_deprecated_pr(
        self,
        new_models: List[Dict],
        deprecated_models: List[Dict],
        failed_models: List[Dict] = None
    ) -> dict [L283-L458]
    calls: get_manifest_path, _find_model_folder_by_dpk_name, _get_file_content, _update_bumpversion_cfg, _update_dataloop_cfg, _generate_unified_pr_title, _generate_unified_pr_body
  ._generate_unified_pr_title(self, new_models: List[Dict], deprecated_models: List[Dict]) -> str [L460-L474]
  ._generate_unified_pr_body(
        self,
        new_models: List[Dict],
        deprecated_models: List[Dict],
        failed_models: List[Dict]
    ) -> str [L476-L536]
  .check_pr_exists(self, model_id: str = None, branch_prefix: str = "nim/") -> Optional[str] [L542-L561]
  .check_model_exists(self, model_id: str, model_type: str) -> bool [L563-L581]
    calls: get_manifest_path, _get_file_content, parse_model_id
  .close_pr(self, pr_number: int, comment: str = None) -> bool [L583-L593]


## agent/license_scraper.py (8 symbols)

fn _normalize_space(text: str) -> str [L91-L92]
fn _clean_html(text: str) -> str [L95-L99]
  calls: _normalize_space
fn _normalize_license(raw: str) -> str [L102-L120]
fn _extract_license_from_text(text: str) -> Optional[str] [L123-L165]
  calls: _normalize_space
fn _fetch_modelcard_sections(model_name: str, publisher: str) -> tuple[str, str] [L171-L273]
  calls: _clean_html
fn _llm_extract_license(model_name: str, page_section: str, api_key: str) -> Optional[str] [L324-L355]
fn find_license(
    model_name: str,
    publisher: str,
    use_llm: bool = True,
    api_key: str = None,
) -> Optional[str] [L361-L410]
  calls: _fetch_modelcard_sections, _llm_extract_license, _normalize_license, _extract_license_from_text
fn find_license_for_resource(resource: dict, use_llm: bool = True, api_key: str = None) -> Optional[str] [L413-L436]
  calls: find_license

## agent/nim_agent.py (33 symbols)

fn _fetch_catalog_by_nim_type(nim_type_filter: str) -> list[dict] [L40-L90]
fn is_model_downloadable(model_name: str) -> bool [L92-L106]
  calls: _normalize_nim_name, _fetch_catalog_by_nim_type
fn get_all_catalog_models(skip_licenses: bool = False) -> list[dict] [L108-L135]
  calls: _fetch_catalog_by_nim_type, find_license_for_resource
fn get_model_ids(models: list[dict]) -> list[str] [L138-L153]
fn get_openai_nim_models(api_key: str = None) -> list[dict] [L156-L191]
fn get_openai_model_ids(api_key: str = None) -> list[str] [L194-L202]
  calls: get_openai_nim_models
fn get_all_repository_models() -> list[dict] [L205-L244]
  calls: load
fn _normalize_nim_name(name: str) -> str [L247-L251]
fn get_repository_downloadable_models() -> list[dict] [L254-L277]
  calls: _normalize_nim_name, _fetch_catalog_by_nim_type, get_all_repository_models
fn update_support_matrix() -> str [L280-L430]
  calls: get_all_repository_models, _normalize_nim_name
fn featch_report() -> dict [L433-L545]
  calls: get_openai_model_ids, _fetch_catalog_by_nim_type
class NIMAgent [L551-L1632]
  """Agent that manages NVIDIA NIM model discovery and onboarding.
    
    Flow:
   ..."""
  .__init__(self, test_project_id: str = None, tester_auto_init: bool = True) [L567-L607]
    calls: Tester, DPKGeneratorClient
  ._get_github(self) -> GitHubClient [L609-L613]
    calls: GitHubClient
  ._get_project(self) -> dl.Project [L615-L619]
  .fetch_models(self, skip_licenses: bool = False) [L625-L686]
    calls: get_openai_nim_models, get_all_catalog_models
  .compare(self) -> dict [L693-L785]
  .fetch_dataloop_dpks(self) -> list [L787-L841]
    calls: _find_nim_dpk, model_name_from_downloadable_dpk_name
  ._normalize(self, name: str) -> str [L843-L845]
  ._extract_model_key(self, name: str) -> str [L847-L869]
  ._models_match(self, name1: str, name2: str) -> bool [L871-L885]
    calls: _extract_model_key
  .onboard_api_model(self, model_id: str, skip_adapter_test: bool = False) -> dict [L891-L950]
    calls: test_single_model, save_manifest_to_repo
  .onboard_api_models(
        self, 
        models: list = None, 
        limit: int = None,
        max_workers: int = 10,
        skip_adapter_test: bool = False,
    ) -> list [L952-L1033]
  ._resolve_downloadable_relative_path(self, model: dict) -> str | None [L1039-L1054]
    calls: _normalize_nim_name, get_all_repository_models
  .onboard_downloadable_models(
        self,
        models: list = None,
        limit: int = None,
        skip_docker: bool = False,
    ) -> list [L1056-L1160]
    calls: _resolve_downloadable_relative_path, build_downloadable_nim
  .preview_downloadables(self, limit: int = None) [L1162-L1187]
    calls: _resolve_downloadable_relative_path
  .open_new_and_deprecated_pr(self) -> dict [L1193-L1254]
    calls: _get_github, create_new_and_deprecated_pr
  .generate_report(self) -> dict [L1261-L1294]
  .print_report(self) [L1296-L1330]
    calls: generate_report
  .save_results(self, output_dir: str = "agent/run_data") [L1332-L1348]
    calls: generate_report
  .run(
        self,
        limit: int = None,
        open_pr: bool = True,
        max_workers: int = 10,
        skip_docker: bool = False,
    ) [L1354-L1406]
    calls: fetch_models, fetch_dataloop_dpks, compare, onboard_api_models, onboard_downloadable_models, update_support_matrix, open_new_and_deprecated_pr, print_report, save_results, generate_report
  .run_agentic(
        self,
        limit: int = None,
        open_pr: bool = True,
        max_workers: int = 10,
        skip_docker: bool = False,
        state_path: str = None,
        downloadable_preview: bool = False,
    ) -> dict [L1412-L1595]
    calls: RunState, load, start_run, get_quarantined, fetch_models, fetch_dataloop_dpks, compare, end_run, save, _write_step_summary, pick_probe_sample, onboard_api_models, preview_downloadables, onboard_downloadable_models, record_result, clear_quarantine, classify_error, open_new_and_deprecated_pr, print_report, save_results, generate_report
  @staticmethod
  ._write_step_summary(run_record: dict, state) [L1598-L1632]
    calls: get_quarantined


## agent/nim_tester.py (25 symbols)

class Tester [L66-L1539]
  """Centralized testing for NVIDIA NIM models.
    
    Tests:
    1. API call - det..."""
  .__init__(self, api_key: str = None, dpk_mcp=None, auto_init: bool = True) [L84-L109]
    calls: _init_test_resources
  ._init_test_resources(self) [L115-L158]
    calls: _create_test_items, _create_test_models
  ._create_test_items(self, dataset) [L160-L195]
    calls: _get_or_create_prompt_item, _get_or_create_vlm_image, _create_vlm_prompt_item, _get_or_create_vlm_video, _create_vlm_video_prompt_item
  ._get_or_create_prompt_item(self, dataset, folder: str, name: str, prompt_content: str) -> dl.Item [L197-L219]
  ._create_vlm_prompt_item(self, image_item: dl.Item, dataset: dl.Dataset) -> dl.Item [L221-L248]
  ._create_vlm_video_prompt_item(self, video_item: dl.Item, dataset: dl.Dataset) -> dl.Item [L250-L279]
  ._get_or_create_vlm_video(self, dataset) -> dl.Item [L281-L311]
    calls: _create_synthetic_video
  ._create_synthetic_video(self) -> str [L313-L358]
  ._get_or_create_vlm_image(self, dataset) -> dl.Item [L360-L390]
  ._get_response_from_item(self, item: dl.Item, model_name: str = None) -> str [L392-L417]
  ._find_nim_dpk(self, nlp: str = None) -> tuple[list[dl.Dpk], str] [L419-L453]
  ._create_test_models(self, project) [L455-L494]
    calls: _find_nim_dpk, _get_or_create_model
  ._get_or_create_model(self, project: dl.Project, config: dict) -> dl.Model [L496-L582]
  .get_test_item_id(self, model_type: str) -> str [L704-L722]
    calls: _init_test_resources
  .get_test_model_id(self, model_type: str) -> str [L724-L742]
    calls: _init_test_resources
  .get_project_id(self) -> str [L596-L600]
    calls: _init_test_resources
  .detect_model_type(self, model_id: str) -> dict [L606-L698]
  ._prepare_model_entity(self, model_type: str, nim_model_id: str, embeddings_size: int = None, nim_invoke_url: str = None) [L744-L759]
    calls: get_test_model_id
  .test_model_adapter(self, nim_model_id: str, model_type: str, embeddings_size: int = None, nim_invoke_url: str = None) -> dict [L761-L862]
    calls: _prepare_model_entity, get_test_item_id, get_adapter_path, _get_response_from_item
  .publish_and_test_dpk(
        self,
        dpk_name: str,
        manifest: dict,
        model_type: str,
        cleanup: bool = True
    ) -> dict [L868-L1081]
    calls: get_project_id, _cleanup_dpk_and_app, get_test_item_id, embed, predict, _get_response_from_item
  ._cleanup_dpk_and_app(self, project: dl.Project, app=None, dpk=None) [L1083-L1157]
  .save_manifest_to_repo(self, model_id: str, model_type: str, manifest: dict) -> str [L1163-L1189]
    calls: get_model_folder
  .post_release_platform_test(self, percentage: float = 0.10) -> dict [L1195-L1395]
    calls: get_project_id, get_test_item_id, embed, predict, _get_response_from_item
  .test_single_model(
        self,
        model_id: str,
        test_platform: bool = False,
        cleanup: bool = True,
        save_manifest: bool = True,
        skip_adapter_test: bool = False,
        license: str = None,
    ) [L1401-L1539]
    calls: detect_model_type, test_model_adapter, DPKGeneratorClient, create_nim_dpk_manifest, publish_and_test_dpk, save_manifest_to_repo


## agent/run_state.py (17 symbols)

fn classify_error(error: str) -> str [L26-L47]
class RunState [L50-L260]
  """Persisted state across NIM Agent runs."""
  .__init__(self, path: str = None) [L53-L59]
  .load(self) [L65-L72]
    calls: load
  .save(self) [L74-L78]
  ._get_model(self, model_id: str) -> dict [L84-L96]
  .record_result(self, model_id: str, status: str, error: str = None) [L98-L130]
    calls: _get_model, classify_error
  .should_attempt(self, model_id: str) -> bool [L132-L149]
  .get_quarantined(self) -> list[str] [L151-L156]
  .pick_probe_sample(self, n: int = None) -> list[str] [L158-L169]
    calls: get_quarantined
  .clear_quarantine(self, model_id: str) [L171-L177]
  .start_run(self) -> dict [L183-L197]
    calls: get_quarantined
  .end_run(self, summary: dict) [L199-L204]
  .last_run_summary(self) -> dict | None [L206-L208]
  @property
  .pr_max_failure_rate(self) -> float [L215-L216]
  @property
  .anomaly_deprecation_threshold(self) -> float [L219-L220]
  .print_status(self) [L226-L260]
    calls: get_quarantined, last_run_summary


## agent/tests/local_agent_debug.py (0 symbols)


## agent/tests/test_agent.py (79 symbols)

class TestNormalizeNimName extends unittest.TestCase [L23-L43]
  """Test _normalize_nim_name used for model comparison."""
  .setUp(self) [L26-L28]
  .test_simple_name(self) [L30-L31]
  .test_publisher_prefix(self) [L33-L34]
  .test_dots_and_dashes(self) [L36-L37]
  .test_already_normalized(self) [L39-L40]
  .test_uppercase(self) [L42-L43]

class TestGetModelIds extends unittest.TestCase [L46-L72]
  """Test get_model_ids which extracts publisher/name IDs."""
  .setUp(self) [L49-L51]
  .test_basic(self) [L53-L59]
    calls: get_model_ids
  .test_missing_publisher_defaults_to_nvidia(self) [L61-L64]
    calls: get_model_ids
  .test_publisher_with_spaces(self) [L66-L69]
    calls: get_model_ids
  .test_empty_list(self) [L71-L72]
    calls: get_model_ids

class TestGetAllCatalogModelsDedup extends unittest.TestCase [L75-L117]
  """Test get_all_catalog_models deduplication logic."""
  @patch("nim_agent.get_downloadable_models")
  @patch("nim_agent.get_api_models")
  .test_dedup_prefers_api(self, mock_api, mock_dl) [L80-L97]
    calls: get_all_catalog_models
  @patch("nim_agent.get_downloadable_models")
  @patch("nim_agent.get_api_models")
  .test_no_duplicates(self, mock_api, mock_dl) [L101-L107]
    calls: get_all_catalog_models
  @patch("nim_agent.get_downloadable_models")
  @patch("nim_agent.get_api_models")
  .test_sorted_by_name(self, mock_api, mock_dl) [L111-L117]
    calls: get_all_catalog_models

class TestFetchCatalogParsing extends unittest.TestCase [L120-L166]
  """Test _fetch_catalog_by_nim_type response parsing."""
  @patch("nim_agent.requests.get")
  .test_parses_resources_and_labels(self, mock_get) [L124-L149]
    calls: _fetch_catalog_by_nim_type
  @patch("nim_agent.requests.get")
  .test_deduplicates_within_page(self, mock_get) [L152-L166]
    calls: _fetch_catalog_by_nim_type

class TestOpenAINimModelsParsing extends unittest.TestCase [L169-L195]
  """Test get_openai_nim_models parsing."""
  @patch("nim_agent.OpenAI")
  .test_parses_model_list(self, mock_openai_cls) [L173-L195]
    calls: get_openai_nim_models

class TestParseModelId extends unittest.TestCase [L202-L226]
  """Test parse_model_id (shared function in dpk_mcp_handler)."""
  .setUp(self) [L205-L207]
  .test_with_publisher(self) [L209-L212]
  .test_without_publisher(self) [L214-L217]
  .test_dots_replaced(self) [L219-L222]
  .test_publisher_dashes_to_underscores(self) [L224-L226]

class TestGetModelFolder extends unittest.TestCase [L229-L251]
  """Test get_model_folder and get_manifest_path (shared functions)."""
  .setUp(self) [L232-L235]
  .test_llm_folder(self) [L237-L239]
  .test_embedding_folder(self) [L241-L243]
  .test_vlm_folder(self) [L245-L247]
  .test_manifest_path(self) [L249-L251]

class TestUpdateBumpversionCfg extends unittest.TestCase [L254-L316]
  """Test _update_bumpversion_cfg add/remove logic."""
  .setUp(self) [L257-L263]
    calls: GitHubClient
  .test_adds_new_entries(self) [L265-L279]
    calls: _update_bumpversion_cfg
  .test_removes_deprecated(self) [L281-L299]
    calls: _update_bumpversion_cfg
  .test_no_duplicate_adds(self) [L301-L316]
    calls: _update_bumpversion_cfg

class TestUpdateDataloopCfg extends unittest.TestCase [L319-L372]
  """Test _update_dataloop_cfg add/remove."""
  .setUp(self) [L322-L328]
    calls: GitHubClient
  .test_adds_and_removes(self) [L330-L348]
    calls: _update_dataloop_cfg
  .test_no_duplicates(self) [L350-L363]
    calls: _update_dataloop_cfg
  .test_handles_invalid_json(self) [L365-L372]
    calls: _update_dataloop_cfg

class TestUnifiedPrTitle extends unittest.TestCase [L375-L406]
  """Test _generate_unified_pr_title."""
  .setUp(self) [L378-L384]
    calls: GitHubClient
  .test_new_only(self) [L386-L391]
    calls: _generate_unified_pr_title
  .test_deprecated_only(self) [L393-L398]
    calls: _generate_unified_pr_title
  .test_both(self) [L400-L406]
    calls: _generate_unified_pr_title

class TestDetectModelType extends unittest.TestCase [L413-L496]
  """Test Tester.detect_model_type heuristics + API call (mocked)."""
  .setUp(self) [L416-L435]
    calls: Tester
  .test_embedding_by_name(self) [L437-L442]
    calls: detect_model_type
  .test_vlm_by_name(self) [L444-L450]
    calls: detect_model_type
  .test_llm_default(self) [L452-L457]
    calls: detect_model_type
  .test_rerank_by_name(self) [L459-L463]
    calls: detect_model_type
  .test_embedding_returns_dimension(self) [L465-L468]
    calls: detect_model_type
  .test_llm_has_no_dimension(self) [L470-L472]
    calls: detect_model_type
  .test_api_error_returns_error_status(self) [L474-L479]
    calls: detect_model_type
  .test_embedding_api_error_returns_error_status(self) [L481-L485]
    calls: detect_model_type
  .test_embedding_dimension_from_api(self) [L487-L496]
    calls: detect_model_type

class TestNormalizeLicense extends unittest.TestCase [L503-L526]
  """Test _normalize_license maps raw strings to canonical names."""
  .setUp(self) [L506-L508]
  .test_apache_variants(self) [L510-L513]
  .test_nvidia_community(self) [L515-L517]
  .test_nvidia_open(self) [L519-L520]
  .test_unknown_returns_other(self) [L522-L523]
  .test_fuzzy_substring_match(self) [L525-L526]

class TestFindLicenseValidation extends unittest.TestCase [L529-L543]
  """Test find_license input validation."""
  .setUp(self) [L532-L534]
  .test_raises_on_empty_publisher(self) [L536-L538]
    calls: find_license
  @patch("license_scraper._fetch_modelcard_sections", return_value=("", "http://example.com"))
  .test_returns_none_on_empty_page(self, mock_fetch) [L541-L543]
    calls: find_license

class TestFindLicenseRegexFallback extends unittest.TestCase [L546-L569]
  """Test find_license regex fallback (no LLM)."""
  .setUp(self) [L549-L551]
  @patch("license_scraper._fetch_modelcard_sections")
  .test_extracts_governed_by(self, mock_fetch) [L554-L560]
    calls: find_license
  @patch("license_scraper._fetch_modelcard_sections")
  .test_extracts_mit(self, mock_fetch) [L563-L569]
    calls: find_license

class TestFindLicenseForResource extends unittest.TestCase [L572-L591]
  """Test find_license_for_resource extracts publisher from labels."""
  .setUp(self) [L575-L577]
  .test_raises_on_missing_name(self) [L579-L581]
  @patch("license_scraper.find_license", return_value="Apache 2.0")
  .test_extracts_publisher_from_labels(self, mock_find) [L584-L591]


## models/api/base_adapter.py (5 symbols)

fn get_downloadable_endpoint_and_cookie(app_id: str) [L28-L46]
class NIMBaseAdapter extends dl.BaseModelAdapter [L49-L149]
  """Base adapter for all NVIDIA NIM model types.

    Handles API/downloadable clien..."""
  .load(self, local_path, **kwargs) [L59-L93]
    calls: get_downloadable_client
  .get_downloadable_client(self, app_id: str) [L95-L122]
    calls: get_downloadable_endpoint_and_cookie
  .check_jwt_expiration(self, margin_seconds: int = 60) [L124-L149]
    calls: get_downloadable_client


## models/api/embeddings/base.py (3 symbols)

class ModelAdapter extends NIMBaseAdapter [L11-L58]
  .call_model_open_ai(self, text) [L13-L26]
  .embed(self, batch, **kwargs) [L28-L58]
    calls: check_jwt_expiration, call_model_open_ai


## models/api/llm/base.py (5 symbols)

class ModelAdapter extends NIMBaseAdapter [L12-L144]
  .prepare_item_func(self, item: dl.Item) [L14-L15]
  ._flatten_messages(self, messages: list[dict], context: str = None) -> list[dict] [L17-L61]
  .call_model(self, messages: list[dict], context: str = None) [L63-L108]
    calls: _flatten_messages
  .predict(self, batch, **kwargs) [L110-L144]
    calls: check_jwt_expiration, call_model


## models/api/object_detection/base.py (8 symbols)

class ModelAdapter extends dl.BaseModelAdapter [L12-L141]
  .load(self, local_path, **kwargs) [L14-L22]
  .prepare_item_func(self, item: dl.Item) [L24-L36]
  .call_model(self, image_b64, payload=None) [L38-L60]
  .extract_annotations_yolox(self, img, image_b64, collection) [L62-L76]
    calls: call_model
  .extract_annotations_paddleocr(self, img, image_b64, collection) [L78-L100]
    calls: call_model
  .extract_annotations_cached(self, item, image_b64, collection) [L102-L124]
    calls: call_model
  .predict(self, batch, **kwargs) [L126-L141]
    calls: extract_annotations_yolox, extract_annotations_paddleocr, extract_annotations_cached


## models/api/vlm/base.py (5 symbols)

class ModelAdapter extends NIMBaseAdapter [L12-L116]
  """VLM adapter for NVIDIA NIM vision models using OpenAI-compatible API (images onl..."""
  .prepare_item_func(self, item: dl.Item) [L21-L22]
  .call_model(self, messages: list[dict]) [L24-L66]
    calls: _prepare_image_content
  .predict(self, batch, **kwargs) [L68-L93]
    calls: check_jwt_expiration, call_model
  ._prepare_image_content(self, messages: list[dict]) -> list[dict] [L95-L116]


## models/downloadable/main.py (5 symbols)

fn _stream_output(pipe, log_level=logging.INFO, prefix="") [L31-L41]
fn _get_gpu_memory() [L44-L74]
  calls: run
class Runner extends dl.BaseServiceRunner [L77-L124]
  """NIM runner that starts the inference server and streams logs.
    
    The serve..."""
  .__init__(self, **kwargs) [L85-L115]
    calls: __init__
  ._monitor_gpu(self) [L117-L124]
    calls: _get_gpu_memory


## models/downloadable/tests/test_simple.py (1 symbols)

fn call_nim_endpoint(app_id: str, endpoint: str, method: str = "get", data: dict = None) [L15-L47]

## nodes/asr/base.py (4 symbols)

class ServiceRunner extends dl.BaseServiceRunner [L12-L78]
  .__init__(self, model_config=None) [L14-L33]
  .audio_transcript(self, item: dl.Item, context: dl.Context) -> dl.Item [L35-L55]
    calls: _transcribe
  ._transcribe(self, audio_path: str) -> str [L57-L78]


## test/__init__.py (0 symbols)


## test/unittests/__init__.py (0 symbols)


## test/unittests/test_baidu_paddleocr.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L24]
  .test_inference(self) [L14-L24]
    calls: load, ModelAdapter, call_model


## test/unittests/test_deepseek_ai_deepseek_r1.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L30]
  .test_inference(self) [L14-L30]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_google_gemma_7b.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_ibm_granite_34b_code_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_meta_llama3_70b_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_meta_llama3_8b_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_meta_llama_3_1_70b_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L25-L42]
  .test_inference(self) [L27-L42]
    calls: load


## test/unittests/test_meta_llama_3_2_90b_vision_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L24-L45]
  .test_inference(self) [L26-L45]
    calls: load


## test/unittests/test_meta_llama_3_3_70b_instruct.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L31-L54]
  .test_inference(self) [L33-L54]
    calls: load, ModelAdapter


## test/unittests/test_microsoft_kosmos_2.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_mistralai_mistral_large.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_nv_yolox_page_elements_v1.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L24]
  .test_inference(self) [L14-L24]
    calls: load, ModelAdapter, call_model


## test/unittests/test_nvidia_neva_22b.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L23]
  .test_inference(self) [L14-L23]
    calls: load, ModelAdapter, call_model_open_ai


## test/unittests/test_university_at_buffalo_cached.py (2 symbols)

class TestModelAdapter extends unittest.TestCase [L12-L29]
  .test_inference(self) [L14-L29]
    calls: load, ModelAdapter, call_model


## to_delete/deploy.py (4 symbols)

fn clean(dpk_name: str) [L26-L42]
fn create_manifest(model_name: str, image_version: str = "0.1.13") -> dict [L45-L71]
  calls: _get_agent_dir
fn publish_and_install(project: dl.Project, manifest: dict, integration_id: str = None) -> dl.App [L74-L123]
  calls: _get_repo_root
fn main() [L126-L165]
  calls: clean, build_docker_image, create_manifest, publish_and_install
