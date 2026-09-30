@api
Feature: Native provider adapters
  The Anthropic, Gemini, and Ollama adapters translate the gateway's canonical
  request into each vendor's own wire format and parse its replies, streams,
  and tool calls back. The fake upstream speaks every one of those formats, so
  these scenarios run the adapters' real translation code end to end.

  Background:
    Given I use the master key

  Scenario Outline: Unary chat through the <provider> adapter
    When I send a chat request for model "<model>" saying "Hello <provider>"
    Then the response status is 200
    And the response header "X-Gateway-Deployment" is "<provider>/<model>"
    And the reply is "Reply to: Hello <provider>"

    Examples:
      | provider  | model          |
      | anthropic | eval-anthropic |
      | gemini    | eval-gemini    |
      | ollama    | eval-ollama    |

  Scenario Outline: Streaming and token usage through the <provider> adapter
    When I stream "<model>" "Count to three" with the OpenAI SDK
    Then the streamed text is "Reply to: Count to three"
    When I ask "<model>" "How many tokens" with the OpenAI SDK
    Then the SDK reports token usage

    Examples:
      | provider  | model          |
      | anthropic | eval-anthropic |
      | gemini    | eval-gemini    |
      | ollama    | eval-ollama    |

  Scenario Outline: Tool calls round-trip through the <provider> adapter
    When I ask "<model>" "__tool__ weather in Paris" with the OpenAI SDK offering the "get_weather" tool
    Then the SDK reply calls the "get_weather" tool
    When I stream "<model>" "__tool__ weather in Rome" with the OpenAI SDK offering the "get_weather" tool
    Then the streamed tool call is "get_weather" with arguments {"city":"hi"}
    When I send "<model>" the result "22C and sunny" of a "get_weather" tool call
    Then the response status is 200
    And the reply is "Tool result: 22C and sunny"

    Examples:
      | provider  | model          |
      | anthropic | eval-anthropic |
      | gemini    | eval-gemini    |
      | ollama    | eval-ollama    |

  Scenario Outline: Upstream failures from the <provider> adapter become a structured error
    When I send a chat request for model "<model>" saying "__fail__ <unique>"
    Then the response status is 502
    And the error code is "all_providers_failed"

    Examples:
      | provider  | model          |
      | anthropic | eval-anthropic |
      | gemini    | eval-gemini    |
      | ollama    | eval-ollama    |

  Scenario Outline: Embeddings through the <provider> adapter
    When I embed "hello world" with "<model>" using the OpenAI SDK
    Then I get 1 embedding of 256 dimensions

    Examples:
      | provider | model             |
      | gemini   | eval-gemini-embed |
      | ollama   | eval-ollama-embed |

  Scenario: Anthropic SDK clients reach a native Anthropic deployment
    When I send "Hi Claude" to "eval-anthropic" with the Anthropic SDK
    Then the Anthropic reply is "Reply to: Hi Claude"
    When I stream "Tell me a story" to "eval-anthropic" with the Anthropic SDK
    Then the streamed text is "Reply to: Tell me a story"

  Scenario: Failover crosses providers, from a dead Anthropic deployment to Gemini
    When I send a chat request for model "eval-multi" saying "hello across vendors"
    Then the response status is 200
    And the response header "X-Gateway-Deployment" is "gemini/eval-multi"
    And the reply is "Reply to: hello across vendors"
