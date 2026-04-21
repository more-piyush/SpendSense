<?php

/**
 * CategoryController.php
 * Copyright (c) 2020 james@firefly-iii.org
 *
 * This file is part of Firefly III (https://github.com/firefly-iii).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

declare(strict_types=1);

namespace FireflyIII\Api\V1\Controllers\Autocomplete;

use FireflyIII\Api\V1\Controllers\Controller;
use FireflyIII\Api\V1\Requests\Autocomplete\AutocompleteApiRequest;
use FireflyIII\Enums\UserRoleEnum;
use FireflyIII\Models\Category;
use FireflyIII\Repositories\Category\CategoryRepositoryInterface;
use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

/**
 * Class CategoryController
 */
final class CategoryController extends Controller
{
    private CategoryRepositoryInterface $repository;
    protected array $acceptedRoles = [UserRoleEnum::READ_ONLY];

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

    private function prependPrediction(array $filtered, Request $request): array
    {
        $servingUrl  = rtrim((string) config('services.spendsense.serving_url', ''), '/');
        $description = trim((string) $request->query('description', ''));
        $amount      = $request->query('amount');

        if ('' === $servingUrl || '' === $description || !is_numeric((string) $amount)) {
            return $filtered;
        }

        $payload = [
            'transaction_id' => (string) Str::uuid(),
            'description'    => $description,
            'amount'         => (float) $amount,
            'currency'       => $this->normalizeOptionalString($request->query('currency')),
            'country'        => $this->normalizeOptionalString($request->query('country')),
            'user_id'        => null !== $request->user() ? (string) $request->user()->id : null,
        ];

        try {
            $response = $this->servingClient($servingUrl)->post('predict/categorization', ['json' => $payload]);
            $status   = $response->getStatusCode();
            $rawBody  = (string) $response->getBody();
            $body     = '' === $rawBody ? [] : json_decode($rawBody, true);

            if ($status >= 400 || !is_array($body) || true === ($body['abstained'] ?? false)) {
                return $filtered;
            }

            $top = $body['predicted_categories'][0] ?? null;
            if (!is_array($top) || '' === trim((string) ($top['category'] ?? ''))) {
                return $filtered;
            }

            $predictedName = trim((string) $top['category']);
            foreach ($filtered as $item) {
                if ($predictedName === ($item['name'] ?? '')) {
                    return $filtered;
                }
            }

            array_unshift($filtered, [
                'id'                  => 'predicted',
                'name'                => $predictedName,
                'prediction'          => true,
                'prediction_confidence' => $top['confidence'] ?? null,
            ]);

            return $filtered;
        } catch (GuzzleException $e) {
            Log::warning('SpendSense category prediction failed during autocomplete.', [
                'message' => $e->getMessage(),
            ]);

            return $filtered;
        }
    }

    /**
     * CategoryController constructor.
     */
    public function __construct()
    {
        parent::__construct();
        $this->middleware(function (Request $request, $next) {
            $this->validateUserGroup($request);
            $this->repository = app(CategoryRepositoryInterface::class);
            $this->repository->setUser($this->user);
            $this->repository->setUserGroup($this->userGroup);

            return $next($request);
        });
    }

    /**
     * Documentation for this endpoint is at:
     * https://api-docs.firefly-iii.org/?urls.primaryName=2.0.0%20(v1)#/autocomplete/getCategoriesAC
     */
    public function categories(AutocompleteApiRequest $request): JsonResponse
    {
        $result   = $this->repository->searchCategory($request->attributes->get('query'), $request->attributes->get('limit'));
        $filtered = $result->map(static fn (Category $item): array => ['id' => (string) $item->id, 'name' => $item->name])->toArray();
        $filtered = $this->prependPrediction($filtered, $request);

        return response()->api($filtered);
    }
}
