# NVIDIA NiM API Model adapters

This repository contains the code for Dataloop model adapters that invoke NVIDIA NIMs served via API

More information on NVIDIA NIMs can be found [here](https://build.nvidia.com/explore/discover).

### Using NVIDIA NIMs in Dataloop Platform

- Select the model you want to use.
- In the `Experience` Tab, click on `Get API Key`

![Get API Key and Endpoint URL](assets/nim_api_key.png)

- Install the model from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace)
- Add the API Key as Secret with name `NGC_API_KEY` to your
  organization's [Data Governance](https://docs.dataloop.ai/docs/overview-1?highlight=data%20governance)
- Add the secret to the model's [service configuration](https://docs.dataloop.ai/docs/service-runtime#secrets-for-faas)

---