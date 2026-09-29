@api
Feature: MCP tool servers
  Model Context Protocol servers are registered once and their tools become
  callable through the gateway.

  Scenario: Register a server and call its tools
    Given I use the master key
    And the fake MCP server is registered
    Then its tools "echo", "add" and "fail" are listed
    When I call its "add" tool with a=2 and b=3
    Then the tool result contains "5"

  Scenario: Tool errors are surfaced, not hidden
    Given I use the master key
    And the fake MCP server is registered
    When I call its "fail" tool
    Then the tool result contains "failed on purpose"

  Scenario: Applications cannot register MCP servers
    Given a virtual key with no limits
    When I try to register an MCP server that runs "/bin/sh"
    Then the response status is 403

  Scenario: Applications can call tools
    Given I use the master key
    And the fake MCP server is registered
    And a virtual key with no limits
    When I call its "echo" tool with text "hello from an app"
    Then the tool result contains "hello from an app"
