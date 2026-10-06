@api
Feature: MCP governance
  Tool use is scoped per key, screened by guardrails, audited, and resilient
  to MCP servers forgetting their sessions.

  Background:
    Given I use the master key
    And the fake MCP server is registered

  Scenario: A key limited to one tool sees and calls only that tool
    Given a virtual key allowed only its "add" MCP tool
    Then the only MCP tool listed is its "add" tool
    When I call its "add" tool with a=2 and b=3
    Then the tool result contains "5"
    When I call its "echo" tool with text "hi"
    Then the tool result contains "not allowed for this API key"

  Scenario: Secrets in tool results are redacted
    When I call its "echo" tool with text "token sk-abcdefghijklmnopqrstuvwxyz0123"
    Then the tool result contains "[API_KEY_REDACTED]"
    And the tool result does not contain "sk-abcdefghijklmnopqrstuvwxyz0123"

  Scenario: Every tool call is audited
    When I call its "add" tool with a=4 and b=5
    Then the tool-call audit log shows an "ok" call to its "add" tool

  Scenario: An expired MCP session is recovered transparently
    When I call its "add" tool with a=1 and b=1
    And the MCP server's session expires
    And I call its "add" tool with a=2 and b=2
    Then the tool result contains "4"
