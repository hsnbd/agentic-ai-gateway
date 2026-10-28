@api
Feature: Key scopes for agentic applications
  An application gives each of its user roles a key. The gateway fills tool
  arguments that identify the user, so the model can never act for someone
  else, and limits which RAG collections each key may read.

  Background:
    Given I use the master key
    And the fake MCP server is registered

  Scenario: A bound argument is hidden from the model and filled from the end user
    Given a virtual key that binds its "echo" tool's "text" argument to the end user
    Then its "echo" tool is listed without a "text" argument
    When I call its "echo" tool with text "someone-else" for end user "u-42"
    Then the tool result contains "u-42"
    And the tool result does not contain "someone-else"
    And the tool-call audit log shows an "ok" call to its "echo" tool

  Scenario: A bound tool needs an end user
    Given a virtual key that binds its "echo" tool's "text" argument to the end user
    When I call its "echo" tool with text "anyone"
    Then the tool result contains "needs the caller's end_user"
    And the tool-call audit log shows an "unbound" call to its "echo" tool

  Scenario: A key sees only the collections it is allowed
    Given a RAG collection named "scoped-public" and one named "scoped-private"
    And a virtual key allowed only the "scoped-public*" collections
    Then that key lists only the "scoped-public" collection
