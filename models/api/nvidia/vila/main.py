import os
import logging
import dtlpy as dl

logger = logging.getLogger(__name__)

class VideoToPrompt:
    @staticmethod
    def convert_video_to_prompt(item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Convert a video item into a prompt item containing the video link.

        Args:
            item: Dataloop video item to convert
            context: Dataloop context object
        Returns:
            dl.Item: A prompt item containing the video link
        """
        node = context.node
        prompt_text = node.metadata['customNodeConfig']["prompt_text"]
        prefix = node.metadata['customNodeConfig']["prefix"]
        directory = node.metadata['customNodeConfig']["directory"]
        if directory is None:
            directory = '/'
        logger.info(directory)
        logger.info(item.name, item.id, item.stream)

        prompt_name = f"{prefix}-{os.path.splitext(item.name)[0]}"
        prompt_item = dl.PromptItem(name=prompt_name)

        # Create prompt name from video name
        new_name = f"{os.path.splitext(item.name)[0]}.json"
        
        # Create prompt item
        prompt_item = dl.PromptItem(name=new_name)
        prompt = dl.Prompt(key="1")
        
        text = f"{prompt_text}\n\nHere is the [video_link]({item.stream})"
        # Add the text element to the prompt
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=text)
        prompt_item.prompts.append(prompt)

        # Upload the prompt item
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=directory, overwrite=True, item_metadata={"user": {"original_item": item.id}})
        # new_item.metadata["user"] = new_item.metadata.get("user", {})
        # new_item.metadata["user"]["original_item"] = item.id
        # new_item = new_item.update()
        logger.info(f"Successfully created prompt item from video: {new_name} from input item {item.id} in directory {directory}")

        return new_item


if __name__ == '__main__':
    """
    Example usage of video to prompt conversion
    """
    # Load environment variables
    import dotenv
    dotenv.load_dotenv()

    # Set up Dataloop environment
    ENV = 'prod'
    DATASET_NAME = 'prompts testing'
    ITEM_ID = '67f3b298baf34cfad702420b'
    dl.setenv(ENV)
    
    # Get project and dataset
    project = dl.projects.get(project_name="Model mgmt demo")
    dataset = project.datasets.get(dataset_name=DATASET_NAME)

    # Example: Convert a video item to a prompt item
    video_item = dataset.items.get(item_id=ITEM_ID)  # Example video item ID

    class DummyNode:
        def __init__(self, metadata):
            self.metadata = metadata

    class DummyContext:
        def __init__(self):
            self.node = None

    context = DummyContext()
    context.node = DummyNode(metadata={'customNodeConfig': {'prompt_text': 'Describe what is happening in the video below.', 'prefix': 'vila', 'directory': None}})

    prompt_item = VideoToPrompt.convert_video_to_prompt(item=video_item, context=context)
    
    print(f"\n\nCreated prompt item: {prompt_item}\n\n")
