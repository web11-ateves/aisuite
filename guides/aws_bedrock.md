# AWS Bedrock

To use AWS Bedrock with `aisuite` you will need to create an AWS account and
navigate to `https://us-west-2.console.aws.amazon.com/bedrock/home`. This route
will be redirected to your default region. In this example the region has been set to
`us-west-2`. Anywhere that is listed can be replaced with your desired region.

Navigate to the `[overview](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/overview)` page
directly or by clicking on the `Get started` link.

## Foundation Model Access

You will first need to give your AWS account access to the foundation models by
visiting the `[modelaccess](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)`
page to enable the models you would like to use. Make note of the `Model ID` for the model you would like to use,
this will be used when using the chat completion example below.

Once that has been enabled set your Access Key and Secret in the env variables:

```shell
export AWS_ACCESS_KEY="your-access-key"
export AWS_SECRET_KEY="your-secret-key"
export AWS_REGION_NAME="region-name"
```
## Create a Chat Completion

Install the boto3 client using your package installer

Example with pip
```shell
pip install boto3
```

Example with poetry
```shell
poetry add boto3
```

In your code:
```python
import aisuite as ai
client = ai.Client()


model_id = "meta.llama3-1-405b-instruct-v1:0" # Mode ID from above
provider = "aws-bedrock"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).





