<?php

declare(strict_types=1);

namespace FireflyIII\Http\Controllers\Transaction;

use FireflyIII\Http\Controllers\Controller;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

final class PredictionController extends Controller
{
    private function normalizeOptionalString(mixed $value): string
    {
        if (!is_string($value)) {
            return '';
        }

        return trim($value);
    }

    private function servingClient(string $servingUrl): Client
    {
        return new Client([
            'base_uri'        => $servingUrl.'/',
            'connect_timeout' => 1.5,
            'timeout'         => (float) config('services.spendsense.timeout', 3.0),
            'http_errors'     => false,
        ]);
    }

    public function category(Request $request): JsonResponse
    {
        $servingUrl = rtrim((string) config('services.spendsense.serving_url', ''), '/');
        if ('' === $servingUrl) {
            return response()->json([
                'enabled'              => false,
                'prediction_available' => false,
                'predicted_categories' => [],
            ]);
        }

        $validated = $request->validate([
            'description' => 'required|string|max:1024',
            'amount'      => 'nullable|numeric',
            'currency'    => 'nullable|string|max:16',
            'country'     => 'nullable|string|max:8',
        ]);

        $payload = [
            'transaction_id' => (string) Str::uuid(),
            'description'    => $validated['description'],
            'amount'         => (float) ($validated['amount'] ?? 0.0),
            'currency'       => $this->normalizeOptionalString($validated['currency'] ?? ''),
            'country'        => $this->normalizeOptionalString($validated['country'] ?? ''),
            'user_id'        => null !== $request->user() ? (string) $request->user()->id : null,
        ];

        try {
            $client = $this->servingClient($servingUrl);

            $response = $client->post('predict/categorization', ['json' => $payload]);
            $status   = $response->getStatusCode();
            $rawBody  = (string) $response->getBody();
            $body     = [] === $rawBody ? [] : json_decode($rawBody, true);

            if ($status >= 400 || !is_array($body)) {
                Log::warning('SpendSense category prediction returned unexpected response.', [
                    'status' => $status,
                    'body'   => $rawBody,
                ]);

                return response()->json([
                    'enabled'              => true,
                    'prediction_available' => false,
                    'predicted_categories' => [],
                ]);
            }

            $predictions = is_array($body['predicted_categories'] ?? null) ? $body['predicted_categories'] : [];
            $top         = $predictions[0] ?? null;

            return response()->json([
                'enabled'              => true,
                'prediction_available' => is_array($top),
                'predicted_categories' => $predictions,
                'suggested_category'   => $top['category'] ?? null,
                'confidence'           => $top['confidence'] ?? null,
                'max_confidence'       => $body['max_confidence'] ?? null,
                'abstained'            => (bool) ($body['abstained'] ?? false),
                'transaction_id'       => $body['transaction_id'] ?? $payload['transaction_id'],
                'model_family'         => $body['model_family'] ?? null,
                'model_version'        => $body['model_version'] ?? null,
                'timestamp'            => $body['timestamp'] ?? null,
                'inference_time_ms'    => $body['inference_time_ms'] ?? null,
            ]);
        } catch (GuzzleException $e) {
            Log::warning('SpendSense category prediction failed.', [
                'message' => $e->getMessage(),
            ]);

            return response()->json([
                'enabled'              => true,
                'prediction_available' => false,
                'predicted_categories' => [],
            ]);
        }
    }

    public function feedback(Request $request): JsonResponse
    {
        $servingUrl = rtrim((string) config('services.spendsense.serving_url', ''), '/');
        if ('' === $servingUrl) {
            return response()->json([
                'enabled' => false,
                'status'  => 'skipped',
            ]);
        }

        $validated = $request->validate([
            'task'                       => 'required|string|in:categorization',
            'transaction_id'             => 'nullable|string|max:128',
            'user_id'                    => 'nullable|string|max:128',
            'model_family'               => 'required|string|max:128',
            'model_version'              => 'required|string|max:128',
            'action'                     => 'required|string|in:accepted,overridden,rejected,dismissed,confirmed',
            'predicted_value'            => 'required|array',
            'predicted_value.category'   => 'nullable|string|max:255',
            'predicted_value.confidence' => 'nullable|numeric',
            'final_value'                => 'nullable|array',
            'final_value.category'       => 'nullable|string|max:255',
            'metadata'                   => 'nullable|array',
            'timestamp'                  => 'required|string|max:128',
        ]);
        $validated['user_id'] = null !== $request->user() ? (string) $request->user()->id : ($validated['user_id'] ?? null);

        try {
            $client = $this->servingClient($servingUrl);

            $response = $client->post('feedback', ['json' => $validated]);
            $status   = $response->getStatusCode();
            $rawBody  = (string) $response->getBody();
            $body     = [] === $rawBody ? [] : json_decode($rawBody, true);

            if ($status >= 400 || !is_array($body)) {
                Log::warning('SpendSense feedback logging returned unexpected response.', [
                    'status' => $status,
                    'body'   => $rawBody,
                ]);

                return response()->json([
                    'enabled' => true,
                    'status'  => 'failed',
                ], 502);
            }

            return response()->json($body);
        } catch (GuzzleException $e) {
            Log::warning('SpendSense feedback logging failed.', [
                'message' => $e->getMessage(),
            ]);

            return response()->json([
                'enabled' => true,
                'status'  => 'failed',
            ], 502);
        }
    }
}
