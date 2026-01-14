/*-------------------------------------------------------------------------
 *
 * slack.go
 *    Slack connector implementation
 *
 * Provides Slack integration for messages and channels.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/connectors/slack.go
 *
 *-------------------------------------------------------------------------
 */

package connectors

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

/* SlackConnector implements ReadWriteConnector for Slack */
type SlackConnector struct {
	token    string
	endpoint string
	client   *http.Client
}

/* slackAPIResponse represents a generic Slack API response */
type slackAPIResponse struct {
	OK      bool   `json:"ok"`
	Error   string `json:"error,omitempty"`
	Warning string `json:"warning,omitempty"`
}

/* slackAuthTestResponse represents auth.test API response */
type slackAuthTestResponse struct {
	slackAPIResponse
	URL     string `json:"url"`
	Team    string `json:"team"`
	User    string `json:"user"`
	TeamID  string `json:"team_id"`
	UserID  string `json:"user_id"`
	BotID   string `json:"bot_id,omitempty"`
	IsBot   bool   `json:"is_bot,omitempty"`
}

/* slackConversationsListResponse represents conversations.list API response */
type slackConversationsListResponse struct {
	slackAPIResponse
	Channels []slackChannel `json:"channels"`
	ResponseMetadata struct {
		NextCursor string `json:"next_cursor"`
	} `json:"response_metadata"`
}

/* slackChannel represents a Slack channel */
type slackChannel struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

/* slackConversationsHistoryResponse represents conversations.history API response */
type slackConversationsHistoryResponse struct {
	slackAPIResponse
	Messages []slackMessage `json:"messages"`
	HasMore  bool           `json:"has_more"`
	ResponseMetadata struct {
		NextCursor string `json:"next_cursor"`
	} `json:"response_metadata"`
}

/* slackMessage represents a Slack message */
type slackMessage struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	User      string `json:"user"`
	TS        string `json:"ts"`
	ThreadTS string `json:"thread_ts,omitempty"`
}

/* slackChatPostMessageResponse represents chat.postMessage API response */
type slackChatPostMessageResponse struct {
	slackAPIResponse
	Channel string `json:"channel"`
	TS      string `json:"ts"`
	Message struct {
		Text string `json:"text"`
		User string `json:"user"`
		TS   string `json:"ts"`
	} `json:"message"`
}

/* NewSlackConnector creates a new Slack connector */
func NewSlackConnector(config Config) (*SlackConnector, error) {
	endpoint := "https://slack.com/api"
	if config.Endpoint != "" {
		endpoint = config.Endpoint
	}

	if config.Token == "" {
		return nil, fmt.Errorf("Slack token is required")
	}

	return &SlackConnector{
		token:    config.Token,
		endpoint: strings.TrimSuffix(endpoint, "/"),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

/* Type returns the connector type */
func (s *SlackConnector) Type() string {
	return "slack"
}

/* makeRequest makes a POST request to Slack API */
func (s *SlackConnector) makeRequest(ctx context.Context, method string, params url.Values) (*http.Response, error) {
	apiURL := fmt.Sprintf("%s/%s", s.endpoint, method)
	
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, strings.NewReader(params.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Authorization", "Bearer "+s.token)

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}

	return resp, nil
}

/* Connect establishes connection using auth.test endpoint */
func (s *SlackConnector) Connect(ctx context.Context) error {
	params := url.Values{}
	resp, err := s.makeRequest(ctx, "auth.test", params)
	if err != nil {
		return fmt.Errorf("Slack connection test failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Slack connection test failed with status %d", resp.StatusCode)
	}

	var authResp slackAuthTestResponse
	if err := json.NewDecoder(resp.Body).Decode(&authResp); err != nil {
		return fmt.Errorf("failed to decode auth response: %w", err)
	}

	if !authResp.OK {
		return fmt.Errorf("Slack authentication failed: %s", authResp.Error)
	}

	return nil
}

/* Close closes the connection */
func (s *SlackConnector) Close() error {
	return nil
}

/* Health checks connection health */
func (s *SlackConnector) Health(ctx context.Context) error {
	return s.Connect(ctx)
}

/* Read reads messages from Slack channel using conversations.history */
func (s *SlackConnector) Read(ctx context.Context, path string) (io.Reader, error) {
	/* Path format: channel_id or channel_name */
	channelID := path
	if !strings.HasPrefix(channelID, "C") && !strings.HasPrefix(channelID, "D") && !strings.HasPrefix(channelID, "G") {
		/* Assume it's a channel name, try to resolve it */
		channels, err := s.List(ctx, "")
		if err != nil {
			return nil, fmt.Errorf("failed to resolve channel name: %w", err)
		}
		
		/* Find channel by name */
		found := false
		for _, ch := range channels {
			if strings.TrimPrefix(ch, "#") == channelID {
				/* Extract channel ID from format "#channel_name (C123456)" or just "C123456" */
				parts := strings.Fields(ch)
				if len(parts) > 0 {
					channelID = strings.Trim(parts[len(parts)-1], "()")
					found = true
					break
				}
			}
		}
		
		if !found {
			return nil, fmt.Errorf("channel not found: %s", path)
		}
	}

	params := url.Values{}
	params.Set("channel", channelID)
	params.Set("limit", "100")

	var allMessages []slackMessage
	cursor := ""

	for {
		if cursor != "" {
			params.Set("cursor", cursor)
		}

		resp, err := s.makeRequest(ctx, "conversations.history", params)
		if err != nil {
			return nil, fmt.Errorf("failed to read messages: %w", err)
		}

		var historyResp slackConversationsHistoryResponse
		if err := json.NewDecoder(resp.Body).Decode(&historyResp); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("failed to decode history response: %w", err)
		}
		resp.Body.Close()

		if !historyResp.OK {
			return nil, fmt.Errorf("Slack API error: %s", historyResp.Error)
		}

		allMessages = append(allMessages, historyResp.Messages...)

		if !historyResp.HasMore {
			break
		}

		cursor = historyResp.ResponseMetadata.NextCursor
		if cursor == "" {
			break
		}
	}

	/* Format messages as JSON */
	messageData, err := json.Marshal(allMessages)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal messages: %w", err)
	}

	return bytes.NewReader(messageData), nil
}

/* Write writes a message to Slack channel using chat.postMessage */
func (s *SlackConnector) Write(ctx context.Context, path string, data io.Reader) error {
	/* Path format: channel_id or channel_name */
	channelID := path
	if !strings.HasPrefix(channelID, "C") && !strings.HasPrefix(channelID, "D") && !strings.HasPrefix(channelID, "G") {
		/* Assume it's a channel name, try to resolve it */
		channels, err := s.List(ctx, "")
		if err != nil {
			return fmt.Errorf("failed to resolve channel name: %w", err)
		}
		
		/* Find channel by name */
		found := false
		for _, ch := range channels {
			if strings.TrimPrefix(ch, "#") == channelID {
				/* Extract channel ID from format "#channel_name (C123456)" or just "C123456" */
				parts := strings.Fields(ch)
				if len(parts) > 0 {
					channelID = strings.Trim(parts[len(parts)-1], "()")
					found = true
					break
				}
			}
		}
		
		if !found {
			return fmt.Errorf("channel not found: %s", path)
		}
	}

	/* Read message content from data */
	var messageText strings.Builder
	if data != nil {
		if _, err := io.Copy(&messageText, data); err != nil {
			return fmt.Errorf("failed to read message data: %w", err)
		}
	}

	if messageText.Len() == 0 {
		return fmt.Errorf("message content is empty")
	}

	params := url.Values{}
	params.Set("channel", channelID)
	params.Set("text", messageText.String())

	resp, err := s.makeRequest(ctx, "chat.postMessage", params)
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Slack API returned status %d", resp.StatusCode)
	}

	var postResp slackChatPostMessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&postResp); err != nil {
		return fmt.Errorf("failed to decode post message response: %w", err)
	}

	if !postResp.OK {
		return fmt.Errorf("Slack API error: %s", postResp.Error)
	}

	return nil
}

/* List lists channels using conversations.list */
func (s *SlackConnector) List(ctx context.Context, path string) ([]string, error) {
	params := url.Values{}
	params.Set("types", "public_channel,private_channel")
	params.Set("exclude_archived", "true")
	params.Set("limit", "200")

	var allChannels []slackChannel
	cursor := ""

	for {
		if cursor != "" {
			params.Set("cursor", cursor)
		}

		resp, err := s.makeRequest(ctx, "conversations.list", params)
		if err != nil {
			return nil, fmt.Errorf("failed to list channels: %w", err)
		}

		var listResp slackConversationsListResponse
		if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("failed to decode list response: %w", err)
		}
		resp.Body.Close()

		if !listResp.OK {
			return nil, fmt.Errorf("Slack API error: %s", listResp.Error)
		}

		allChannels = append(allChannels, listResp.Channels...)

		cursor = listResp.ResponseMetadata.NextCursor
		if cursor == "" {
			break
		}
	}

	/* Format channel list */
	result := make([]string, len(allChannels))
	for i, ch := range allChannels {
		result[i] = fmt.Sprintf("#%s (%s)", ch.Name, ch.ID)
	}

	return result, nil
}
